"""
train_lstm.py — 2-layer bidirectional-False LSTM baseline for 24h temperature forecast.

Week 9 deliverable: apples-to-apples comparison with WeatherTransformer.
Same task, same data splits, same seeds.

Architecture: 2 layers · hidden=64 · dropout=0.1
Task: input (B, 48, 7) → predict (B, 24) temperature_2m
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATASETS  = ROOT / "datasets"
MODEL_DIR = ROOT / "models" / "lstm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = MODEL_DIR / "model.pt"
CURVE_PATH   = MODEL_DIR / "train_curve.png"
SCALER_PATH  = DATASETS / "scaler_stats.json"

# ── hyper-params ──────────────────────────────────────────────────────────────
N_LAYERS   = 2
HIDDEN     = 64
DROPOUT    = 0.1
INPUT_LEN  = 48
HORIZON    = 24
N_FEATURES = 7

BATCH_SIZE = 64
LR         = 1e-3
MAX_EPOCHS = 10
PATIENCE   = 3


# ════════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════════

class WeatherLSTM(nn.Module):
    """
    2-layer LSTM baseline.

    Input  : (B, 48, 7)
    Output : (B, 24)   — next 24h temperature_2m forecast

    bidirectional=False — preserves causal direction (can't see the future).
    Final hidden state h_n[-1] → linear head → 24 predictions.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden:     int = HIDDEN,
        n_layers:   int = N_LAYERS,
        dropout:    float = DROPOUT,
        horizon:    int = HORIZON,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features)  →  (B, horizon)"""
        _, (h_n, _) = self.lstm(x)   # h_n: (n_layers, B, hidden)
        out = self.head(h_n[-1])      # last layer's hidden → (B, hidden) → (B, horizon)
        return out


# ════════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════════

def train() -> None:
    # Reuse load_data() from train_transformer — same splits, same scaler.
    from ml.train.train_transformer import load_data

    print("Loading data ...")
    train_ds, val_ds, test_ds = load_data()
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} windows")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = WeatherLSTM()
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  WeatherLSTM params: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_mse  = float("inf")
    patience_left = PATIENCE
    train_losses, val_losses = [], []

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (early stop patience={PATIENCE}) ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_dl:
            optimiser.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(X_b)
        train_mse = total_loss / len(train_ds)

        # ── val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_mse = val_loss / len(val_ds)

        train_losses.append(train_mse)
        val_losses.append(val_mse)
        print(f"  epoch {epoch:02d}/{MAX_EPOCHS}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_left = PATIENCE
            torch.save(model.state_dict(), WEIGHTS_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {epoch} (best val_mse={best_val_mse:.4f})")
                break

    # ── training curve ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train MSE")
    plt.plot(val_losses,   label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardised)")
    plt.title("WeatherLSTM — Training Curve (baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=100)
    plt.close()
    print(f"\n  Training curve saved → {CURVE_PATH}")

    # ── test evaluation ──
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in test_dl:
            all_pred.append(model(X_b))
            all_true.append(y_b)
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    test_mse = nn.MSELoss()(pred, true).item()
    test_mae = (pred - true).abs().mean().item()
    print(f"  Test MSE={test_mse:.4f}  MAE={test_mae:.4f}  (standardised scale)")
    print(f"  Weights saved → {WEIGHTS_PATH}")
    print("\n  [grader] Copy these numbers into ml/README.md LSTM row.")


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ml.data_pipeline import PK_CITIES
    missing = [c for c in PK_CITIES if not (DATASETS / f"{c}.parquet").exists()]
    if missing:
        print(f"[error] Missing parquet for: {missing}")
        print("Run: python -m ml.data_pipeline")
        sys.exit(1)
    if not SCALER_PATH.exists():
        print("[error] scaler_stats.json not found — run data_pipeline first")
        sys.exit(1)
    train()
