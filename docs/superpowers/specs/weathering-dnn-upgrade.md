# Weathering With You — Approach B DNN Upgrade Spec

**Course:** AI341 Deep Neural Networks — GIK Institute  
**Date:** 2026-04-28  
**Status:** Approved (Sections 1–5)

---

## Section 1 — Architecture Overview

### Approach B: 7 PyTorch DNN Modules as Agent Tools

GNN and DQN are explicitly deferred. All remaining modules are wired into the FastAPI + Groq agentic loop.

```
Browser ──WebSocket /ws/chat──► FastAPI (main.py)
                                     │
                               agent.run_agent()
                                     │
                         Groq llama-3.3-70b-versatile
                         (agentic while-loop: calls tools until no tool_calls remain)
                                     │
          ┌──────────────────────────┴──────────────────────────────┐
          │ Open-Meteo tools (tools.py)                             │
          │  get_current_weather, get_hourly_forecast,              │
          │  get_7day_forecast, get_aqi, get_historical_weather,    │
          │  get_historical_hourly, get_weather_alerts,             │
          │  get_uv_index                                           │
          ├─────────────────────────────────────────────────────────┤
          │ ML inference tools (ml/infer/*.py)                      │
          │  transformer_forecast  → transformer_infer.py           │
          │  retrieve_weather_context → rag_infer.py                │
          │  detect_weather_anomaly → vae_infer.py                  │
          │  classify_sky_image → clip_infer.py                     │
          └─────────────────────────────────────────────────────────┘
                                     │
                          response text + ZERO_STATE JSON
                                     │
                        ┌────────────┴────────────┐
                        │  POST /api/tts          │
                        │  edge-tts → mp3 bytes   │
                        └─────────────────────────┘
```

### Module Priority Table

| Module | Train script | Infer wrapper | Wired into Zero? | Syllabus week |
|---|---|---|---|---|
| `data_pipeline` | (is the pipeline) | — | No | 5 |
| `transformer_forecaster` | `train_transformer.py` | `transformer_infer.py` | Yes | 8 |
| `lstm_forecaster` | `train_lstm.py` | `lstm_infer.py` | No (baseline) | 9 |
| `text_embeddings_rag` | FAISS at startup | `rag_infer.py` | Yes | 10 |
| `cnn_sky_classifier` | `train_cnn.py` | `cnn_infer.py` | No (standalone) | 11 |
| `vae_anomaly` | `train_vae.py` | `vae_infer.py` | Yes | 12 |
| `clip_sky_check` | zero-shot | `clip_infer.py` | Yes | 12 |

### lifespan() Startup Sequence (main.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load transformer weights → transformer_infer.MODEL
    # 2. Load LSTM weights → lstm_infer.MODEL
    # 3. Load CNN weights → cnn_infer.MODEL
    # 4. Load VAE weights → vae_infer.MODEL
    # 5. Build FAISS index → rag_infer.INDEX
    # 6. Load CLIP model → clip_infer.MODEL (zero-shot, no weights file)
    # Each wrapped in try/except — missing .pt → graceful error dict, never crash
    yield
    # teardown (nothing needed)
```

---

## Section 2 — ML Modules (Data + Models)

### Module 1: data_pipeline.py

**Source:** Open-Meteo Archive API — `https://archive-api.open-meteo.com/v1/archive`  
(NOT `/forecast` — that endpoint only goes ~92 days back)

**Cities:** Karachi, Lahore, Islamabad, Topi, Peshawar, Quetta (6 PK cities from `tools.py:PK_CITIES`)

**Features fetched hourly:**
`temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`, `weather_code`, `surface_pressure`, `cloud_cover`, `apparent_temperature`, `uv_index`

**Time split (time-series discipline — never random-split):**
- Train: 2023 full year
- Val: 2024 Jan–Jun
- Test: 2024 Jul–Dec

**Quality:** NaN rate < 1% per feature; forward-fill imputation ≤ 3 hours gap; `imputation_log.json` records all fills.

**Output:** `ml/datasets/{city}.parquet` + `scaler_stats.json` (μ/σ per feature)

### Module 2: Transformer Forecaster (from scratch — Week 8 deliverable)

**Constraint:** NO `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`. Must implement from primitives.

```python
class MultiHeadSelfAttention(nn.Module):
    # Explicit Q, K, V: nn.Linear(d_model, d_model)
    # scaled dot-product: (Q @ K.T) / sqrt(d_k)
    # softmax → dropout → weighted sum → output projection

class PositionalEncoding(nn.Module):
    # sin/cos formula; no nn.Embedding

class TransformerEncoderBlock(nn.Module):
    # MHSA → residual → LayerNorm → FFN → residual → LayerNorm

class WeatherTransformer(nn.Module):
    # 4 blocks, 4 heads, d_model=128
    # Input: (B, seq=24, features=9) → Output: (B, 6)  # next 6h temp
```

**Saves:** `ml/models/transformer/transformer.pt` + `attn_sample.npy` (for viva heatmap)

### Module 3: LSTM Forecaster (baseline, Week 9)

```python
class WeatherLSTM(nn.Module):
    # 2 layers, hidden=64, input=9 features, output=6h forecast
    # bidirectional=False (keeps causal direction)
```

**Saves:** `ml/models/lstm/lstm.pt`

### Module 4: RAG — Text Embeddings + FAISS (Week 10)

- Model: `sentence-transformers/all-MiniLM-L6-v2` (22M params, CPU-fast)
- Index type: `faiss.IndexFlatIP` (inner product = cosine after L2-normalising embeddings)
- Corpus: `ml/datasets/weather_snippets.json` — 200 curated weather event descriptions for PK cities
- Built at server startup via `rag_infer.build_index()`
- Tool: `retrieve_weather_context(query: str) → list[str]` — top-3 passages

### Module 5: CNN Sky Classifier (Week 11)

- Architecture: ResNet-style, ~5 blocks, <2M params, `CrossEntropyLoss` (no `WeightedRandomSampler`)
- Labels: `["clear", "partly_cloudy", "overcast", "rain", "fog", "sunset"]` (6 classes)
- Input: 224×224 RGB, standard ImageNet normalisation
- **Saves:** `ml/models/cnn/cnn.pt`

### Module 6: VAE Anomaly Detector (Week 12)

- Architecture: encoder/decoder 2 layers each, `latent=8`
- Loss: β-annealed ELBO — β: 0 → 0.5 over first 10 epochs (prevents posterior collapse)
- Input: **parquet weather vectors** (NOT text snippets — RAG owns the text corpus)
- Reconstruction error > μ+3σ → anomaly flag
- **Saves:** `ml/models/vae/vae.pt`

### Module 7: CLIP Zero-Shot Sky Classifier (Week 12)

- Model: `openai/clip-vit-base-patch32` (no fine-tuning)
- 6 text prompts → map to 3-label CNN set via:

```python
def map_to_cnn_labels(clip_label: str) -> str:
    mapping = {
        "clear sky": "clear",
        "partly cloudy sky": "partly_cloudy",
        "overcast sky": "overcast",
        "rainy sky": "rain",
        "foggy sky": "fog",
        "sunset sky": "overcast",   # sunset → overcast (conservative)
    }
    return mapping.get(clip_label, "clear")
```

- Tool: `classify_sky_image(image_b64: str) → {clip_label, cnn_label, confidence}`

---

## Section 3 — Voice Agent (TTS) Design

### Stack

| Layer | Technology |
|---|---|
| Primary TTS | `edge-tts` (Microsoft Azure neural via Edge free endpoint) |
| Emotion clips | Bark (one-shot, pre-rendered to `.wav`, never at request time) |
| Fallback TTS | `pyttsx3` (most-female SAPI voice, never male) |
| Audio concat | `pydub` |
| Language detect | `langdetect` (per sentence) |
| Frontend playback | `useRef`-based `<audio>` element, no `speechSynthesis` |

### Voice Routing

| Detected language | Voice |
|---|---|
| Urdu / Roman-Urdu | `ur-PK-UzmaNeural` |
| English | `en-GB-SoniaNeural` (preferred) / `en-US-JennyNeural` (fallback) |

Mixed responses → split on language boundary → concatenate audio chunks.

### Inline Emotion Tags

| Tag | Behavior |
|---|---|
| `[sigh]` | 400ms pre-rendered Bark breath clip |
| `[yawn]` | 600ms yawn clip |
| `[cry]` | 500ms soft sob clip |
| `[gasp]` | 200ms inhale clip |
| `[pause]` | SSML `<break time="400ms"/>` |
| `[whisper]…[/whisper]` | SSML `<prosody rate="slow" pitch="-10%" volume="x-soft">` |
| `[loud]…[/loud]` | SSML `<prosody rate="fast" pitch="+5%" volume="loud">` |
| `[excited]…[/excited]` | SSML `<prosody rate="fast" pitch="+15%" volume="medium">` |
| `[soft]…[/soft]` | SSML `<prosody rate="slow" pitch="-5%" volume="soft">` |

### Trigger Guidance (SYSTEM_PROMPT)

- Rain forecast → `[excited]` wrap, occasional `[gasp]`
- Heat warning → `[sigh]` prefix, `[soft]` body
- Late-night queries (00:00–05:00) → `[yawn]` prefix, `[whisper]…[/whisper]`
- Severe alert → `[loud]` wrap
- Sad/bad AQI → `[soft]` or `[cry]` (once per conversation max)
- Default → no tags, neutral delivery

### Hard Rules

- Always female. Never male. Never `hi-IN-*` (Hindi) voices.
- No paid TTS. `edge-tts` unreachable → `pyttsx3` female fallback.
- All synthesis is server-side. `useVoice.js` only plays blobs from `/api/tts`.

---

## Section 4 — Implementation Details

### synth.py Flow

```
synthesize(text: str) -> bytes
  1. Strip ZERO_STATE suffix
  2. Parse inline tags → list[TaggedSegment]
  3. For each segment:
     a. If clip tag ([sigh]/[yawn]/[cry]/[gasp]): load pre-rendered wav
     b. If text segment: langdetect → pick voice → build SSML → edge-tts.Communicate.stream()
  4. pydub concat all segments → export MP3 → return bytes
  5. On edge-tts failure: pyttsx3 fallback (asyncio.to_thread for blocking call)
```

### useVoice.js Pattern

```js
const audioRef = useRef(null);
const objectUrlRef = useRef(null);

async function speak(text) {
  try {
    stop();
    const res = await fetch('/api/tts', { method:'POST', body: JSON.stringify({text}),
                                          headers:{'Content-Type':'application/json'} });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    objectUrlRef.current = url;
    const audio = new Audio(url);
    audioRef.current = audio;
    audio.onended = () => { URL.revokeObjectURL(url); objectUrlRef.current = null; };
    audio.play().catch(() => {});   // Chrome autoplay policy
  } catch {
    if (objectUrlRef.current) { URL.revokeObjectURL(objectUrlRef.current); objectUrlRef.current = null; }
  }
}

function stop() {
  if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
  if (objectUrlRef.current) { URL.revokeObjectURL(objectUrlRef.current); objectUrlRef.current = null; }
}
```

### /api/tts Endpoint

```python
class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    audio = await synthesize(req.text)
    return Response(content=audio, media_type="audio/mpeg")
```

### asyncio Note

`pydub` and `pyttsx3` are blocking. Wrap with `asyncio.to_thread()` inside `async synthesize()`.

---

## Section 5 — File Map & Requirements

### Complete File Tree

```
backend/
├── main.py                          # + lifespan(), /api/tts
├── agent.py                         # + 4 ML tool defs, emotion-tag SYSTEM_PROMPT
├── tools.py                         # unchanged
├── cache.py                         # unchanged
├── requirements.txt                 # + TTS + ML deps
├── tts/
│   ├── __init__.py
│   ├── synth.py                     # async synthesize()
│   ├── generate_clips.py            # one-time Bark runner
│   └── clips/                       # sigh.wav yawn.wav cry.wav gasp.wav
└── ml/
    ├── __init__.py
    ├── bootstrap.py                 # runs all train scripts in order
    ├── README.md                    # grader table + run instructions
    ├── data_pipeline.py
    ├── datasets/
    │   ├── karachi.parquet  lahore.parquet  islamabad.parquet
    │   ├── topi.parquet  peshawar.parquet  quetta.parquet
    │   ├── scaler_stats.json
    │   ├── weather_snippets.json
    │   └── imputation_log.json
    ├── models/
    │   ├── transformer/  (transformer.pt, attn_sample.npy)
    │   ├── lstm/         (lstm.pt)
    │   ├── cnn/          (cnn.pt)
    │   └── vae/          (vae.pt)
    ├── train/
    │   ├── train_transformer.py
    │   ├── train_lstm.py
    │   ├── train_cnn.py
    │   └── train_vae.py
    ├── infer/
    │   ├── transformer_infer.py
    │   ├── lstm_infer.py
    │   ├── rag_infer.py
    │   ├── vae_infer.py
    │   ├── clip_infer.py
    │   └── cnn_infer.py
    └── notebooks/
        ├── 01_data_pipeline.ipynb
        ├── 02_transformer.ipynb
        ├── 03_lstm.ipynb
        ├── 04_rag.ipynb
        ├── 05_cnn.ipynb
        ├── 06_vae.ipynb
        └── 07_clip.ipynb
```

### requirements.txt Additions

```
# TTS
edge-tts>=6.1.9
pydub>=0.25.1
langdetect>=1.0.9
pyttsx3>=2.90
# suno-bark           # uncomment only to run tts/generate_clips.py — heavy download

# ML
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pandas>=2.2.0
pyarrow>=15.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
Pillow>=10.3.0
numpy>=1.26.0
jupyter>=1.0.0
tensorboard>=2.16.0
```

`torch-geometric` absent — GNN deferred.

### bootstrap.py Order

```python
# Order matters: data must exist before models train on it
run("python -m ml.data_pipeline")
run("python -m ml.train.train_transformer")   # ~8–12 min
run("python -m ml.train.train_lstm")          # ~3–5 min
run("python -m ml.train.train_cnn")           # ~4–6 min
run("python -m ml.train.train_vae")           # ~3–5 min
# RAG index and CLIP load at server startup — no offline training needed
```

Wall-clock budget: ~20–31 min total on 8-core CPU laptop.

### Wall-Clock Budget

| Step | Estimate |
|---|---|
| data_pipeline | 2–4 min |
| train_transformer | 8–12 min |
| train_lstm | 3–5 min |
| train_cnn | 4–6 min |
| train_vae | 2–3 min |
| **Total** | **~19–30 min** |

---

## Implementation Tracks

### Track A — Voice/TTS

| Step | Action |
|---|---|
| A1 | Append TTS deps to `backend/requirements.txt` |
| A2 | Create `backend/tts/__init__.py` and `backend/tts/synth.py` |
| A3 | Create `backend/tts/generate_clips.py` |
| A4 | Add `/api/tts` POST endpoint to `backend/main.py` |
| A5 | Rewrite `frontend/src/hooks/useVoice.js` |
| A6 | Update SYSTEM_PROMPT in `backend/agent.py` with emotion-tag vocabulary |
| A7 | Test TTS round-trip |
| A8 | Run `python -m tts.generate_clips` once (manual — Bark ~2GB download) |

### Track B — ML Layer

| Step | Action |
|---|---|
| B1 | Create `backend/ml/` directory tree + `__init__.py` files |
| B2 | Write `data_pipeline.py` |
| B3 | Write `train/train_transformer.py` (from-scratch attention) |
| B4 | Write `train/train_lstm.py` |
| B5 | Write `train/train_cnn.py` |
| B6 | Write `train/train_vae.py` |
| B7 | Write all 6 infer wrappers |
| B8 | Write `bootstrap.py` |
| B9 | Wire 4 ML tools into `agent.py` |
| B10 | Update `lifespan()` in `main.py` |
| B11 | Run `python -m ml.bootstrap` |
