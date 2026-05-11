"""
TTS synthesis pipeline for Zero.

Flow: strip ZERO_STATE → parse inline tags → per-sentence langdetect →
      build SSML → edge-tts stream → splice Bark clips → pydub concat → MP3 bytes.

Blocking calls (pydub, pyttsx3) are wrapped in asyncio.to_thread.
"""

import asyncio
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIPS_DIR = Path(__file__).parent / "clips"

# Clip durations are informational only — actual wav files drive timing.
CLIP_FILES = {
    "sigh": CLIPS_DIR / "sigh.wav",
    "yawn": CLIPS_DIR / "yawn.wav",
    "cry":  CLIPS_DIR / "cry.wav",
    "gasp": CLIPS_DIR / "gasp.wav",
}

# Roman-Urdu trigger words that force Urdu voice even when langdetect sees English.
ROMAN_URDU_TRIGGERS = {
    "yaar", "bohat", "subhanallah", "subhan'allah", "tufaan", "bilkul",
    "acha", "achha", "theek", "matlab", "waisay", "waise", "zyada",
    "baarish", "garmi", "sardi", "mausam", "araam", "mushkil", "khair",
    "haan", "nahi", "nahin", "bas", "uff", "arey", "arre", "mashallah",
}

VOICE_URDU   = "ur-PK-UzmaNeural"
VOICE_EN_GB  = "en-GB-SoniaNeural"
VOICE_EN_US  = "en-US-JennyNeural"   # fallback if SoniaNeural unavailable

# ZERO_STATE marker — strip everything from here to end of string.
ZERO_STATE_RE = re.compile(r"ZERO_STATE:\{.*?\}$", re.DOTALL)

# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

# Matches [tag], [/tag], or [tag]...[/tag] blocks.
_TAG_RE = re.compile(
    r"\[(?P<close>/)?(?P<name>sigh|yawn|cry|gasp|pause|whisper|loud|excited|soft)\]"
)

# Span tags that wrap content (need open+close).
_SPAN_TAGS = {"whisper", "loud", "excited", "soft"}
# Point tags that insert a clip or break (no close tag).
_POINT_TAGS = {"sigh", "yawn", "cry", "gasp", "pause"}

SSML_PROSODY = {
    "whisper":  'rate="slow" pitch="-10%" volume="x-soft"',
    "loud":     'rate="fast" pitch="+5%" volume="loud"',
    "excited":  'rate="fast" pitch="+15%" volume="medium"',
    "soft":     'rate="slow" pitch="-5%" volume="soft"',
}


@dataclass
class TaggedSegment:
    kind: str           # "text" | "clip" | "break"
    text: str = ""      # for kind="text": the sentence content (may have SSML)
    clip_name: str = "" # for kind="clip": key into CLIP_FILES
    ssml_attrs: str = ""# for kind="text": any wrapping prosody attrs
    lang: str = "en"    # detected language ("ur" or "en")


def _detect_lang(sentence: str) -> str:
    """Return 'ur' if sentence is Urdu/Roman-Urdu, else 'en'."""
    # Check for Arabic script first (real Urdu).
    if re.search(r"[؀-ۿ]", sentence):
        return "ur"
    # Check for Roman-Urdu trigger words (case-insensitive).
    words = set(re.findall(r"[a-zA-Z']+", sentence.lower()))
    if words & ROMAN_URDU_TRIGGERS:
        return "ur"
    # Fall back to langdetect; treat any South Asian language as Urdu rather
    # than letting it bleed into Hindi.
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        detected = detect(sentence)
        if detected in ("ur", "hi"):   # hi → treat as ur (never use Hindi voice)
            return "ur"
    except Exception:
        pass
    return "en"


def _voice_for_lang(lang: str) -> str:
    return VOICE_URDU if lang == "ur" else VOICE_EN_GB


def _parse_segments(raw: str) -> list[TaggedSegment]:
    """
    Convert raw LLM text (with inline tags) into a flat list of TaggedSegment.

    Strategy: walk the string, collecting text between tags.  When a span-open
    tag is seen, collect text until its close tag and mark it with SSML attrs.
    Point tags immediately emit a clip/break segment.
    """
    segments: list[TaggedSegment] = []
    pos = 0
    text_buf = ""
    active_span: Optional[str] = None  # current open span tag name, if any

    def flush_text(buf: str, span: Optional[str] = None) -> None:
        """Split buf into sentences and emit a TaggedSegment per sentence."""
        if not buf.strip():
            return
        # Split on sentence boundaries while preserving trailing punctuation.
        sentences = re.split(r"(?<=[.!?؟])\s+", buf.strip())
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            lang = _detect_lang(s)
            attrs = SSML_PROSODY.get(span, "") if span else ""
            segments.append(TaggedSegment(kind="text", text=s, ssml_attrs=attrs, lang=lang))

    for m in _TAG_RE.finditer(raw):
        # Accumulate text before this tag.
        text_buf += raw[pos:m.start()]
        pos = m.end()

        tag_name  = m.group("name")
        is_close  = bool(m.group("close"))

        if tag_name in _POINT_TAGS and not is_close:
            flush_text(text_buf, active_span)
            text_buf = ""
            if tag_name == "pause":
                segments.append(TaggedSegment(kind="break"))
            else:
                segments.append(TaggedSegment(kind="clip", clip_name=tag_name))

        elif tag_name in _SPAN_TAGS:
            if not is_close:
                # Opening span: flush prior text without this span's attrs.
                flush_text(text_buf, active_span)
                text_buf = ""
                active_span = tag_name
            else:
                # Closing span: flush text WITH active span's attrs.
                flush_text(text_buf, active_span)
                text_buf = ""
                active_span = None

    # Remaining text after last tag.
    text_buf += raw[pos:]
    flush_text(text_buf, active_span)

    return segments


# ---------------------------------------------------------------------------
# edge-tts synthesis
# ---------------------------------------------------------------------------

async def _synthesize_segment_edge(segment: TaggedSegment) -> bytes:
    """Synthesize one text segment via edge-tts, returning raw MP3 bytes."""
    import edge_tts

    voice = _voice_for_lang(segment.lang)

    if segment.ssml_attrs:
        ssml = (
            "<speak>"
            f'<prosody {segment.ssml_attrs}>'
            f"{segment.text}"
            "</prosody>"
            "</speak>"
        )
        communicate = edge_tts.Communicate(ssml, voice, rate="+0%")
    else:
        communicate = edge_tts.Communicate(segment.text, voice)

    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# pyttsx3 fallback (blocking — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _pyttsx3_synthesize_blocking(text: str) -> bytes:
    """Fallback: use pyttsx3 with the most female-sounding SAPI voice."""
    import pyttsx3
    import tempfile

    engine = pyttsx3.init()

    # Pick the most female-sounding voice available; never default to a male.
    voices = engine.getProperty("voices")
    female_voice = None
    for v in voices:
        name_lower = (v.name or "").lower()
        if any(k in name_lower for k in ("zira", "hazel", "heather", "susan",
                                          "female", "woman", "girl", "sonia",
                                          "jenny", "aria", "uzma")):
            female_voice = v.id
            break
    if female_voice is None and voices:
        # Last resort: first voice in list (still better than crashing).
        female_voice = voices[0].id

    if female_voice:
        engine.setProperty("voice", female_voice)

    engine.setProperty("rate", 165)
    engine.setProperty("volume", 0.9)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# pydub audio assembly (blocking — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _assemble_audio_blocking(
    segment_data: list[tuple[str, bytes | None]]
) -> bytes:
    """
    Concatenate audio chunks and return MP3 bytes.

    Fast path: if all chunks are MP3 (no WAV clips, no breaks), concatenate
    raw bytes directly. MP3 is a streaming format — frame-level concatenation
    produces valid audio without ffmpeg.

    Slow path (requires ffmpeg via pydub): any WAV clip or silence break
    triggers full decode → mix → re-encode. Falls back to raw concatenation
    if pydub/ffmpeg is unavailable, so the server never crashes.
    """
    needs_pydub = any(kind in ("wav", "break") for kind, _ in segment_data)

    if not needs_pydub:
        # Fast path: raw MP3 frame concatenation — no ffmpeg needed.
        return b"".join(data for _, data in segment_data if data)

    # Slow path: mix MP3 + WAV + silence via pydub (requires ffmpeg).
    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        silence_400ms = AudioSegment.silent(duration=400)

        for kind, data in segment_data:
            if kind == "break":
                combined += silence_400ms
            elif kind == "mp3" and data:
                combined += AudioSegment.from_file(io.BytesIO(data), format="mp3")
            elif kind == "wav" and data:
                combined += AudioSegment.from_file(io.BytesIO(data), format="wav")

        out = io.BytesIO()
        combined.export(out, format="mp3")
        return out.getvalue()

    except Exception:
        # ffmpeg not installed — fall back to raw MP3 concatenation only
        # (WAV clips and breaks are dropped; voice still plays).
        return b"".join(data for kind, data in segment_data
                        if kind == "mp3" and data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def synthesize(text: str) -> bytes:
    """
    Convert LLM text (with inline tags) to MP3 bytes.

    Steps:
      1. Strip ZERO_STATE suffix.
      2. Parse inline tags → TaggedSegment list.
      3. For each segment: synthesize text via edge-tts (fallback: pyttsx3),
         load pre-rendered clip for point tags, insert silence for [pause].
      4. Assemble with pydub (blocking → asyncio.to_thread).
      5. Return MP3 bytes.
    """
    # 1. Strip ZERO_STATE.
    clean = ZERO_STATE_RE.sub("", text).strip()
    if not clean:
        return b""

    # 2. Parse.
    segments = _parse_segments(clean)
    if not segments:
        return b""

    # 3. Synthesize each segment concurrently where possible.
    #    Clips are loaded from disk (fast); text goes to edge-tts.
    raw_chunks: list[tuple[str, bytes | None]] = []

    for seg in segments:
        if seg.kind == "break":
            raw_chunks.append(("break", None))

        elif seg.kind == "clip":
            clip_path = CLIP_FILES.get(seg.clip_name)
            if clip_path and clip_path.exists():
                wav_bytes = clip_path.read_bytes()
                raw_chunks.append(("wav", wav_bytes))
            # If clip file missing (generate_clips.py not yet run), skip silently.

        elif seg.kind == "text":
            mp3_bytes: bytes = b""
            try:
                mp3_bytes = await _synthesize_segment_edge(seg)
            except Exception:
                # edge-tts failure → pyttsx3 fallback (blocking).
                try:
                    mp3_bytes = await asyncio.to_thread(
                        _pyttsx3_synthesize_blocking, seg.text
                    )
                except Exception:
                    pass  # if both fail, skip segment rather than crash
            if mp3_bytes:
                raw_chunks.append(("mp3", mp3_bytes))

    if not raw_chunks:
        return b""

    # 4. Assemble with pydub in a thread (blocking).
    result = await asyncio.to_thread(_assemble_audio_blocking, raw_chunks)
    return result
