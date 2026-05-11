"""
One-time Bark emotion clip generator.

Run once before first server start:
    python -m tts.generate_clips

Writes four WAV files to backend/tts/clips/:
    sigh.wav   (~400ms breath)
    yawn.wav   (~600ms yawn)
    cry.wav    (~500ms soft sob)
    gasp.wav   (~200ms sharp inhale)

Bark is imported lazily — this module can be imported on a server that does
NOT have suno-bark installed without raising ImportError.  Only running
__main__ requires the package.

Speaker preset: v2/en_speaker_9  (female, breathy — best for emotion clips).
"""

from pathlib import Path
import numpy as np

CLIPS_DIR = Path(__file__).parent / "clips"

# Each entry: (filename, prompt, trim_ms)
# trim_ms: keep only this many milliseconds of audio (Bark often pads silence).
CLIP_SPECS = [
    ("sigh.wav",  "[sighs softly]",           400),
    ("yawn.wav",  "[yawns quietly]",           600),
    ("cry.wav",   "[cries softly, sniffles]",  500),
    ("gasp.wav",  "[gasps sharply]",           200),
]

SPEAKER_PRESET = "v2/en_speaker_9"   # female, breathy
SAMPLE_RATE    = 22050               # Bark's native output rate


def _trim_to_ms(audio: np.ndarray, sr: int, ms: int) -> np.ndarray:
    """Keep only the first `ms` milliseconds of a mono audio array."""
    samples = int(sr * ms / 1000)
    return audio[:samples] if len(audio) > samples else audio


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    from pydub import AudioSegment

    # Bark outputs float32 in [-1, 1]; convert to int16 PCM for pydub.
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    seg = AudioSegment(
        pcm_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,   # 16-bit = 2 bytes
        channels=1,       # mono
    )
    seg.export(str(path), format="wav")


def generate_all(force: bool = False) -> None:
    """Generate all emotion clips. Skips existing files unless force=True."""
    # Lazy import — only needed at generation time, not at server import time.
    try:
        from bark import generate_audio, SAMPLE_RATE as BARK_SR  # type: ignore
        from bark.preload import preload_models                    # type: ignore
    except ImportError:
        raise SystemExit(
            "suno-bark is not installed.\n"
            "Uncomment '# suno-bark' in requirements.txt and run:\n"
            "    pip install suno-bark\n"
            "then re-run: python -m tts.generate_clips"
        )

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Bark models (~2 GB first download, cached after)...")
    preload_models()
    print("Models ready.\n")

    for filename, prompt, trim_ms in CLIP_SPECS:
        out_path = CLIPS_DIR / filename
        if out_path.exists() and not force:
            print(f"  [skip]  {filename} already exists — pass force=True to overwrite")
            continue

        print(f"  [gen]   {filename}  prompt={prompt!r}  target={trim_ms}ms ...")
        audio = generate_audio(prompt, history_prompt=SPEAKER_PRESET)

        # Bark may return at its own sample rate; honour it if different.
        sr = BARK_SR if BARK_SR else SAMPLE_RATE
        audio_trimmed = _trim_to_ms(audio, sr, trim_ms)

        _save_wav(out_path, audio_trimmed, sr)
        actual_ms = int(len(audio_trimmed) / sr * 1000)
        print(f"          saved → {out_path}  ({actual_ms}ms, {sr}Hz mono)\n")

    print("Done. All clips written to", CLIPS_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Bark emotion clips for Zero TTS.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing clip files."
    )
    args = parser.parse_args()

    generate_all(force=args.force)
