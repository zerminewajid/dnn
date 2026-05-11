"""
train_transformer.py — from-scratch Transformer for 24h temperature forecasting.

Week 8 deliverable: MultiHeadSelfAttention built from primitives.
NO nn.Transformer / nn.MultiheadAttention / nn.TransformerEncoderLayer.

Architecture: 4 layers · 4 heads · d_model=128 · FFN=256 · dropout=0.1
Task: input (B, 48, 7) → predict (B, 24) temperature_2m
Data: 6 PK cities, train=2023, val=2024 H1, test=2024 H2
"""

import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent          # backend/ml/
DATASETS  = ROOT / "datasets"
MODEL_DIR = ROOT / "models" / "transformer"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH    = MODEL_DIR / "model.pt"
ATTN_PATH       = MODEL_DIR / "attn_sample.npy"
CURVE_PATH      = MODEL_DIR / "train_curve.png"
SCALER_PATH     = DATASETS / "scaler_stats.json"

# ── hyper-params ──────────────────────────────────────────────────────────────
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
D_FFN      = 256
DROPOUT    = 0.1
INPUT_LEN  = 48
HORIZON    = 24
N_FEATURES = 7      # must match data_pipeline.FEATURES

BATCH_SIZE = 64
LR         = 1e-3
MAX_EPOCHS = 15
PATIENCE   = 3      # early-stop patience on val MSE

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "weather_code", "surface_pressure", "cloud_cover",
]
TEMP_IDX = FEATURES.index("temperature_2m")


# ════════════════════════════════════════════════════════════════════════════════
# Model — all attention from primitives
# ════════════════════════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention — no nn.MultiheadAttention.

    Q, K, V are explicit nn.Linear projections.
    Attention: softmax( Q·Kᵀ / √d_k ) · V
    Output projection: nn.Linear(d_model, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.scale   = math.sqrt(self.d_k)

        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_o  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, d_model)
        returns (out, attn_weights)
          out         : (B, T, d_model)
          attn_weights: (B, n_heads, T, T)
        """
        B, T, _ = x.shape

        # Project and split into heads: (B, n_heads, T, d_k)
        def project_split(W):
            out = W(x)                                       # (B, T, d_model)
            return out.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = project_split(self.W_q)
        K = project_split(self.W_k)
        V = project_split(self.W_v)

        # Scaled dot-product attention
        scores  = (Q @ K.transpose(-2, -1)) / self.scale    # (B, H, T, T)
        weights = torch.softmax(scores, dim=-1)              # (B, H, T, T)
        weights = self.drop(weights)

        # Weighted sum + merge heads
        ctx = weights @ V                                    # (B, H, T, d_k)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        out = self.W_o(ctx)
        return out, weights


class PositionalEncoding(nn.Module):
    """
    Fixed sin/cos positional encoding (Vaswani et al. 2017).
    No nn.Embedding — computed from formula.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN layout: LayerNorm → MHSA → residual → LayerNorm → FFN → residual.
    (Pre-LN trains more stably on small CPU runs than post-LN.)
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (out, attn_weights)."""
        normed       = self.norm1(x)
        attn_out, w  = self.attn(normed)
        x            = x + attn_out           # residual 1
        x            = x + self.ffn(self.norm2(x))   # residual 2
        return x, w


class WeatherTransformer(nn.Module):
    """
    Input  : (B, 48, 7)  — 48h window, 7 features
    Output : (B, 24)     — next 24h temperature forecast

    Param count: ~400 K — well within 8-core CPU budget.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model:    int = D_MODEL,
        n_heads:    int = N_HEADS,
        n_layers:   int = N_LAYERS,
        d_ffn:      int = D_FFN,
        dropout:    float = DROPOUT,
        horizon:    int = HORIZON,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        self.blocks     = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, horizon)   # pooled over time → horizon

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """
        x : (B, T, n_features)
        returns logits (B, horizon) or (logits, [attn per block]) if return_attn
        """
        x = self.input_proj(x)      # (B, T, d_model)
        x = self.pos_enc(x)

        attns = []
        for block in self.blocks:
            x, w = block(x)
            attns.append(w)

        x   = self.norm(x)
        out = self.head(x.mean(dim=1))   # mean-pool over time → (B, d_model) → (B, horizon)

        if return_attn:
            return out, attns
        return out


# ════════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load parquet files for all 6 cities, apply train-scaler, build windows.
    Returns (train_ds, val_ds, test_ds).
    """
    import pandas as pd
    from ml.data_pipeline import (
        PK_CITIES, make_windows, split_df, apply_scaler
    )

    with open(SCALER_PATH) as f:
        stats = json.load(f)

    train_xs, train_ys = [], []
    val_xs,   val_ys   = [], []
    test_xs,  test_ys  = [], []

    for city in PK_CITIES:
        path = DATASETS / f"{city}.parquet"
        if not path.exists():
            print(f"  [skip] {city}.parquet not found — run data_pipeline first")
            continue

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = apply_scaler(df, stats)

        tr, va, te = split_df(df)
        for split_df_, xs, ys in [(tr, train_xs, train_ys),
                                   (va, val_xs,   val_ys),
                                   (te, test_xs,  test_ys)]:
            X, y = make_windows(split_df_, INPUT_LEN, HORIZON)
            if len(X):
                xs.append(X)
                ys.append(y)

    def cat(xs, ys):
        return TensorDataset(torch.cat(xs), torch.cat(ys))

    return cat(train_xs, train_ys), cat(val_xs, val_ys), cat(test_xs, test_ys)


# ════════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════════

def train() -> None:
    print("Loading data ...")
    train_ds, val_ds, test_ds = load_data()
    print(
        f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} windows"
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WeatherTransformer()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  WeatherTransformer params: {n_params:,}")

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
            pred = model(X_b)
            loss = criterion(pred, y_b)
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

    # ── save attention sample ──
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    sample_X = val_ds[0][0].unsqueeze(0)    # (1, 48, 7)
    with torch.no_grad():
        _, attns = model(sample_X, return_attn=True)
    # Save last block, all heads: (n_heads, T, T)
    attn_np = attns[-1].squeeze(0).cpu().numpy()   # (4, 48, 48)
    np.save(ATTN_PATH, attn_np)
    print(f"\n  Attention sample saved → {ATTN_PATH}  shape={attn_np.shape}")

    # ── training curve ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train MSE")
    plt.plot(val_losses,   label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardised)")
    plt.title("WeatherTransformer — Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=100)
    plt.close()
    print(f"  Training curve saved → {CURVE_PATH}")

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
    print(f"\n  Test MSE={test_mse:.4f}  MAE={test_mae:.4f}  (standardised scale)")
    print(f"  Weights saved → {WEIGHTS_PATH}")


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Guard: parquet must exist before training
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
