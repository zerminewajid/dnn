# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Weathering With You** — a Pakistani Gen-Z agentic weather app for AI341 (GIK Institute). The backend is a FastAPI + Groq agentic loop; the frontend is a React/Tailwind/Framer Motion SPA with "Zero", an animated orb bot character. An ML layer (`backend/ml/`) is being added: 7 PyTorch DNN modules that become agent tools.

---

## Dev Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
# Requires GROQ_API_KEY in .env
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # dev server on :5173, proxies /api and /ws to :8000
npm run build      # outputs dist/ for production serving by FastAPI
```

### Docker (full stack)
```bash
docker-compose up --build   # port 7860
```

### ML Training (offline, run once before starting the server)
```bash
cd backend
python -m ml.bootstrap          # runs all train scripts in order
# Or individually:
python -m ml.train.train_transformer
python -m ml.train.train_lstm
python -m ml.train.train_cnn
python -m ml.train.train_vae
```

### TTS Clip Generation (optional — Bark, deferred)
```bash
cd backend
python -m tts.generate_clips    # renders Bark sigh/yawn/cry/gasp wavs to tts/clips/
```
> **Current shipped posture:** Bark clips are **not pre-generated**. The four
> point tags `[sigh]` `[yawn]` `[cry]` `[gasp]` are **silently stripped** at
> synthesis time when `tts/clips/` is empty — `synth.py` checks
> `clip_path.exists()` before loading, so the server never crashes.
> SSML span tags (`[whisper]`, `[loud]`, `[excited]`, `[soft]`) work fully as
> documented (no Bark dependency). A grader who wants the emotion clips can
> run `python -m tts.generate_clips` once — requires ~2 GB Bark download and
> ~10 min CPU generation. The rest of the app is unaffected either way.

---

## Architecture

### Request Flow
```
Browser ──WebSocket /ws/chat──► FastAPI (main.py)
                                     │
                               agent.run_agent()
                                     │
                         Groq llama-3.3-70b-versatile
                         (agentic while-loop: calls tools
                          until no tool_calls remain)
                                     │
               ┌─────────────────────┴──────────────────────┐
               │ Open-Meteo tools (tools.py)                 │
               │ ML inference tools (ml/infer/*.py)          │
               └─────────────────────────────────────────────┘
                                     │
                          response text + ZERO_STATE
                                     │
                        ┌────────────┴────────────┐
                        │  POST /api/tts          │
                        │  edge-tts → mp3 bytes   │
                        └─────────────────────────┘
```

The frontend also calls REST endpoints (`/api/weather/*`) directly for the weather cards — these bypass the agent entirely.

### Key Backend Files

| File | Role |
|---|---|
| `backend/agent.py` | Groq client, TOOLS list, SYSTEM_PROMPT, agentic while-loop, ZERO_STATE parsing |
| `backend/tools.py` | Async Open-Meteo wrappers; `PK_CITIES` dict is the canonical 6-city set |
| `backend/main.py` | FastAPI app, WebSocket `/ws/chat`, REST routes, `/api/tts` endpoint, React build serving |
| `backend/cache.py` | In-memory TTL cache (600s default); Redis-capable via `init_cache(url)` |
| `backend/tts/synth.py` | edge-tts synthesis, per-sentence language detection, SSML + inline-tag handling |
| `backend/tts/clips/` | Pre-rendered Bark emotion clips (sigh/yawn/cry/gasp) |
| `backend/tts/generate_clips.py` | One-time Bark clip generator |

### Adding a New Agent Tool
1. Write the async function in `tools.py` or `ml/infer/*.py`
2. Add its tool definition to the `TOOLS` list in `agent.py` (OpenAI function-calling format)
3. Add a dispatch branch in `_execute_tool()` in `agent.py`
4. Update `SYSTEM_PROMPT` routing guidance if the tool needs explicit Llama routing hints

### ZERO_STATE Contract — DO NOT CHANGE
The agent appends this JSON suffix to every response:
```
ZERO_STATE:{"state":"IDLE|REACTING_RAIN|REACTING_HOT|REACTING_COLD|REACTING_WIND|SPEAKING","temperature":int,"rain_probability":int}
```
`main.py` strips it before sending `text` to the frontend. `useZero.js` reads `bot_state` from the WebSocket message to drive avatar animations. Changing the schema breaks the frontend.

### Frontend Structure

- `App.jsx` — top-level: owns WebSocket connection, weather state, Zero state; wires all hooks and components together
- `components/Zero.jsx` — animated orb with Framer Motion state machine; states map to `useZero.js` BOT_STATE enum
- `hooks/useZero.js` — bot state machine (IDLE, SCANNING, REACTING_*, SPEAKING, SLEEPING, EXPLODING); drives eye color, scale, animation
- `hooks/useWeather.js` — fetches current/hourly/7day/AQI via REST; manages city selection
- `hooks/useVoice.js` — plays audio blobs returned from `/api/tts`. Does NOT call `window.speechSynthesis`.
- `vite.config.js` — proxies `/api` and `/ws` to `:8000` in dev

### ML Layer (`backend/ml/`) — Approach B

Modules in priority order (highest to lowest):

| Module | Train script | Infer wrapper | Wired into Zero? |
|---|---|---|---|
| `data_pipeline.py` | (is the pipeline) | — | No |
| `transformer_forecaster` | `train/train_transformer.py` | `infer/transformer_infer.py` | Yes — `transformer_forecast` tool |
| `lstm_forecaster` | `train/train_lstm.py` | `infer/lstm_infer.py` | No — baseline for comparison |
| `text_embeddings_rag` | FAISS built at startup | `infer/rag_infer.py` | Yes — `retrieve_weather_context` tool |
| `cnn_sky_classifier` | `train/train_cnn.py` | `infer/cnn_infer.py` | No — standalone |
| `vae_anomaly` | `train/train_vae.py` | `infer/vae_infer.py` | Yes — `detect_weather_anomaly` tool |
| `clip_sky_check` | No training (zero-shot) | `infer/clip_infer.py` | Yes — `classify_sky_image` tool |

**Dataset:** 2 years hourly Open-Meteo for 6 PK cities, cached as parquet in `ml/datasets/`. Time split: train=2023, val=2024 Jan–Jun, test=2024 Jul–Dec. Never random-split time-series.

**All train scripts:** `torch.manual_seed(42)`, `np.random.seed(42)`. Models save weights to `ml/models/<name>/`. If weights are missing at startup, infer wrappers return a graceful error dict — Zero relays it in-character, the server never crashes.

### Transformer "From Scratch" Constraint

The Week 8 deliverable requires implementing attention from primitives. In `ml/train/train_transformer.py` and any related code:
- DO NOT use `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoderLayer`.
- Implement `MultiHeadSelfAttention` with explicit Q/K/V `nn.Linear` projections, scaled dot-product, softmax, output projection.
- Implement `PositionalEncoding` with the sin/cos formula.
- Stack into `TransformerEncoderBlock` with residual + LayerNorm + FFN.
- Save attention weights for one validation sample to `ml/models/transformer/attn_sample.npy` for the viva visualization.

### Model Size Budgets (CPU-only target)

All training must finish in <15 min on an 8-core CPU laptop, no GPU. Stay within these caps:

| Model | Cap |
|---|---|
| Transformer | 4 layers, 4 heads, d_model=128 |
| LSTM | 2 layers, hidden=64 |
| CNN | ResNet-style, ~5 blocks, <2M params |
| VAE | latent=8, encoder/decoder 2 layers each |
| DQN (if added later) | 2-layer MLP |

### Time-Series Discipline

- Train: 2023 full year. Val: 2024 Jan–Jun. Test: 2024 Jul–Dec.
- Never random-split. Never let val/test leak into train via overlapping windows.
- Document the split in every notebook's first code cell.

### Notebook Template (apply to all `ml/notebooks/*.ipynb`)

1. Markdown — problem statement + syllabus week.
2. Markdown — math derivation (loss, gradients, key equations).
3. Code — data loading + preprocessing (with shape comments).
4. Code — model definition (shape comment on every tensor).
5. Code — training loop with live loss print.
6. Markdown — architecture diagram (mermaid or matplotlib).
7. Code — evaluation + plots (loss curve, confusion matrix, attention heatmap, etc.).
8. Markdown — comparison vs baseline + reflection on trade-offs.

### Agent Routing Rules (kept in SYSTEM_PROMPT — change there too if you change here)

- "will it rain tomorrow" / multi-hour forecast → `transformer_forecast`
- "is this weather weird/unusual" → `detect_weather_anomaly`
- sky photo upload → `classify_sky_image`
- weather news / event context → `retrieve_weather_context` (RAG, call before answering)
- live current temp/humidity → `get_current_weather` (Open-Meteo)

### Character Voice Guardrails

Zero is autistic-coded, Pakistani, weather-obsessed. **Always female voice.** Short, soft, observational sentences. Excited by rain. Distressed by heat. Never breaks character, even on errors. ≤3 sentences unless user asks for detail.

#### Voice / TTS Stack (free only — no ElevenLabs, no paid APIs)

Primary engine: `edge-tts` (Microsoft Azure neural voices via Edge's free endpoint). Backend module: `backend/tts/synth.py`. Frontend `useVoice.js` plays the returned audio blob via `<audio>` instead of `speechSynthesis`.

Voice routing — **language detected per sentence, voice swapped per sentence**:

| Detected language | Voice | Style |
|---|---|---|
| Urdu (Arabic script, or Roman-Urdu phrases like "yaar", "bohat", "subhan'allah", "tufaan") | `ur-PK-UzmaNeural` | Pakistani female, warm, soft |
| English | `en-GB-SoniaNeural` (preferred) or `en-US-JennyNeural` (fallback) | British female, intelligent, measured |

If a single response mixes both (e.g. "It's 38 degrees, yaar"), split on language boundaries and concatenate the audio chunks — never force one voice to speak the other language.

#### Emotional Cues (SSML + inline tags)

Zero's text from the LLM may contain inline emotion tags. The TTS layer interprets them; they are stripped from the visible chat text:

| Tag | Behavior |
|---|---|
| `[sigh]` | 400ms breath sample (pre-rendered Bark clip), then continue |
| `[yawn]` | 600ms yawn clip |
| `[whisper]…[/whisper]` | SSML `<prosody rate="slow" pitch="-10%" volume="x-soft">` |
| `[loud]…[/loud]` | SSML `<prosody rate="fast" pitch="+5%" volume="loud">` |
| `[cry]` | 500ms soft sob clip |
| `[gasp]` | 200ms inhale clip |
| `[pause]` | SSML `<break time="400ms"/>` |
| `[excited]…[/excited]` | SSML `<prosody rate="fast" pitch="+15%" volume="medium">` |
| `[soft]…[/soft]` | SSML `<prosody rate="slow" pitch="-5%" volume="soft">` |

Pre-rendered emotion clips live in `backend/tts/clips/` (generated once with Bark via `python -m tts.generate_clips`, then `.wav` files committed for fast playback). Do NOT run Bark at request time — it's too slow on CPU.

#### When Zero uses which cue (prompt-side guidance)

The SYSTEM_PROMPT in `agent.py` instructs Llama to emit these tags naturally:
- Rain forecast → `[excited]` wrap, occasional `[gasp]`
- Heat warning → `[sigh]` prefix, `[soft]` body
- Late-night queries (00:00–05:00) → `[yawn]` prefix, `[whisper]…[/whisper]`
- Severe alert → `[loud]` wrap
- Air-quality bad / sad observation → `[soft]` or `[cry]` (sparingly — once per conversation max so it stays meaningful)
- Default observation → no tags, neutral delivery

#### Hard rules

- Voice is female in 100% of outputs. Never fall back to a male voice.
- Pakistani (`ur-PK-UzmaNeural`) is the ONLY Urdu voice. Never use Indian Hindi voices (`hi-IN-*`) even though they sound similar.
- All TTS is server-side. Frontend `useVoice.js` only plays audio blobs returned from `/api/tts`; it does NOT call `window.speechSynthesis` anymore.
- No external paid TTS service may be added. If `edge-tts` is unreachable, fall back to `pyttsx3` with the most-female-sounding installed SAPI voice — never to a male default.
- Audio is streamed where possible (`edge-tts` supports streaming) so Zero starts speaking before the full sentence is synthesized.

### Environment Variables
```
GROQ_API_KEY=...          # required — Groq LLM
REDIS_URL=...             # optional — falls back to in-memory cache
```
Never commit `.env`.
