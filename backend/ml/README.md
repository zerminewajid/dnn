# ML Layer — Weathering With You (AI341)

## How to Run

```bash
cd backend

# Step 0 — install deps
pip install -r requirements.txt

# Step 1 — run all training in order (~20–31 min on 8-core CPU)
python -m ml.bootstrap

# Or individually
python -m ml.data_pipeline
python -m ml.train.train_transformer
python -m ml.train.train_lstm
python -m ml.train.train_cnn
python -m ml.train.train_vae

# Step 2 — start server (RAG + CLIP load at startup)
uvicorn main:app --reload --port 8000
```

## Grader Table

| Module | Syllabus Week | CLO | Architecture | Target Metric | Achieved |
|---|---|---|---|---|---|
| `data_pipeline` | 5 | CLO 2 | Open-Meteo Archive → parquet | NaN rate <1% after ffill(≤3h) | — |
| `transformer_forecaster` | 8 | CLO 3 | 4L · 4H · d=128 · from-scratch MHSA | Test MSE (standardised) | — |
| `lstm_forecaster` | 9 | CLO 3 | 2L · h=64 · causal | Test MSE (baseline) | — |
| `text_embeddings_rag` | 10 | CLO 3 | MiniLM-L6 + FAISS IndexFlatIP | Top-3 retrieval relevance | — |
| `cnn_sky_classifier` | 11 | CLO 3 | ResNet-5 · <2M params · 3-class | Test accuracy | — |
| `vae_anomaly` | 12 | CLO 3 | VAE latent=8 · β-annealed ELBO | Test recon MSE · anomaly % | — |
| `clip_sky_check` | 12 | CLO 3 | CLIP ViT-B/32 zero-shot · 6→3 labels | vs CNN accuracy | — |

> Fill in "Achieved" column after running `python -m ml.bootstrap`.

## Where Outputs Live

```
ml/
├── datasets/
│   ├── {city}.parquet          hourly weather 2023-2024, 6 cities
│   ├── scaler_stats.json       train-split mean/std per feature
│   ├── imputation_log.json     features with >1% NaN after ffill
│   └── weather_snippets.json   RAG corpus (30-50 curated snippets)
└── models/
    ├── transformer/
    │   ├── model.pt            best val-MSE weights
    │   ├── attn_sample.npy     (4, 48, 48) attention from last block, val sample
    │   └── train_curve.png     train vs val MSE per epoch
    ├── lstm/
    │   ├── model.pt
    │   └── train_curve.png
    ├── cnn/
    │   ├── model.pt
    │   ├── train_curve.png     loss (left) + val accuracy (right)
    │   └── confusion_matrix.png
    └── vae/
        ├── model.pt
        ├── anomaly_threshold.json   {"threshold": float, "percentile": 95}
        ├── train_curve.png          recon MSE (left) + KL (right)
        └── latent_scatter.png       PCA 2D of test latents, coloured by recon error
```

## Reading the Curves

- **Transformer / LSTM**: lower val MSE = better. Watch for val > train gap (overfitting).
- **CNN**: val accuracy plateaus when features saturate. Check confusion matrix for class bias.
- **VAE**: recon should decrease. KL should rise slowly with β annealing (0→0.5 over 10 epochs).
  If KL stays at 0 all training, posterior collapsed — re-run with a lower initial LR.

## Visualising Attention (Viva)

```python
import numpy as np, matplotlib.pyplot as plt
attn = np.load("ml/models/transformer/attn_sample.npy")  # (4, 48, 48)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.imshow(attn[i], cmap="viridis", aspect="auto")
    ax.set_title(f"Head {i+1}")
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
plt.suptitle("WeatherTransformer — Attention Weights (last block, val sample)")
plt.tight_layout(); plt.show()
```

## Time Split (NEVER random-split)

| Split | Date range | Rows/city |
|---|---|---|
| Train | 2023-01-01 → 2023-12-31 | ~8,760 |
| Val   | 2024-01-01 → 2024-06-30 | ~4,368 |
| Test  | 2024-07-01 → 2024-12-31 | ~4,416 |

Scaler fit on **train only**. Val and test are never touched during parameter search.

## Agent Tool Routing

| User intent | Tool called |
|---|---|
| "Will it rain tomorrow?" | `transformer_forecast` |
| "Is this weather unusual?" | `detect_weather_anomaly` |
| Sky photo upload | `classify_sky_image` (CLIP) |
| Weather event context | `retrieve_weather_context` (RAG) |
| Current conditions | `get_current_weather` (Open-Meteo) |
