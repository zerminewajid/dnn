"""
train_vae.py — β-annealed Variational Autoencoder for weather anomaly detection.

Week 12 deliverable: detect unusual weather conditions via reconstruction error.

Input : single-timestep standardised 7-feature vector (NOT a sequence)
Output: reconstructed vector + anomaly score (MSE reconstruction error)

Architecture: encoder [7→32→16] → (μ, logσ²) → z (dim=8)
              decoder [8→16→32→7]
β annealing: β=0 → 0.5 linearly over first 10 epochs (prevents posterior collapse).
Anomaly threshold = 95th percentile of train-split reconstruction error.
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
from torch.utils.data import DataLoader, TensorDataset

# ── seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATASETS  = ROOT / "datasets"
MODEL_DIR = ROOT / "models" / "vae"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH   = MODEL_DIR / "model.pt"
THRESHOLD_PATH = MODEL_DIR / "anomaly_threshold.json"
CURVE_PATH     = MODEL_DIR / "train_curve.png"
SCATTER_PATH   = MODEL_DIR / "latent_scatter.png"
SCALER_PATH    = DATASETS  / "scaler_stats.json"

# ── hyper-params ──────────────────────────────────────────────────────────────
N_FEATURES  = 7
LATENT_DIM  = 8
HIDDEN_DIM  = 32       # first hidden layer width
HIDDEN_DIM2 = 16       # second hidden layer width (bottleneck before latent)

BATCH_SIZE  = 128
LR          = 1e-3
MAX_EPOCHS  = 15
BETA_MAX    = 0.5
BETA_ANNEAL = 10       # epochs over which β rises from 0 → BETA_MAX

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "weather_code", "surface_pressure", "cloud_cover",
]

TRAIN_END  = "2023-12-31"
VAL_START  = "2024-01-01"
VAL_END    = "2024-06-30"
TEST_START = "2024-07-01"

ANOMALY_PERCENTILE = 95   # threshold = 95th pct of train recon errors


# ════════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """7 → 32 → 16 → (μ, logσ²) each of dim=latent_dim."""

    def __init__(self, n_features: int = N_FEATURES, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM2),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(HIDDEN_DIM2, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM2, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, logσ²), each (B, latent_dim)."""
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """latent_dim → 16 → 32 → 7 (mirrors encoder)."""

    def __init__(self, latent_dim: int = LATENT_DIM, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, n_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → x_hat: (B, n_features)."""
        return self.net(z)


class WeatherVAE(nn.Module):
    """
    Variational Autoencoder for single-timestep weather feature vectors.

    Reparameterization: z = μ + σ·ε,  ε ~ N(0, I)
    where σ = exp(0.5 · logσ²) for numerical stability.
    """

    def __init__(self, n_features: int = N_FEATURES, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(n_features, latent_dim)
        self.decoder = Decoder(latent_dim, n_features)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """z = μ + σ·ε  (only adds noise during training)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu   # deterministic at eval time

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, μ, logσ²)."""
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decoder(z)
        return x_hat, mu, logvar


# ════════════════════════════════════════════════════════════════════════════════
# Loss
# ════════════════════════════════════════════════════════════════════════════════

def elbo_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO = E[log p(x|z)] − β·KL(q(z|x) ‖ p(z))

    Reconstruction: MSE (feature space, standardised).
    KL: -0.5 · Σ(1 + logσ² − μ² − σ²), summed over latent dims, mean over batch.

    Returns (total_loss, recon_mse, kl) — all scalar tensors.
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return recon + beta * kl, recon, kl


def _beta(epoch: int) -> float:
    """Linear anneal: 0 at epoch 1, BETA_MAX at epoch BETA_ANNEAL+1, constant after."""
    return min(BETA_MAX, BETA_MAX * (epoch - 1) / BETA_ANNEAL)


# ════════════════════════════════════════════════════════════════════════════════
# Data loading — single timestep vectors from parquet
# ════════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load all 6 city parquets, apply train-scaler, return flat feature vectors.
    No windowing — VAE sees individual hourly rows.
    """
    import pandas as pd
    from ml.data_pipeline import PK_CITIES, apply_scaler

    with open(SCALER_PATH) as f:
        stats = json.load(f)

    train_vecs, val_vecs, test_vecs = [], [], []

    for city in PK_CITIES:
        path = DATASETS / f"{city}.parquet"
        if not path.exists():
            print(f"  [skip] {city}.parquet not found — run data_pipeline first")
            continue

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = apply_scaler(df, stats)

        tr = df.loc[df.index <= TRAIN_END, FEATURES].dropna()
        va = df.loc[(df.index >= VAL_START) & (df.index <= VAL_END), FEATURES].dropna()
        te = df.loc[df.index >= TEST_START, FEATURES].dropna()

        train_vecs.append(torch.tensor(tr.values, dtype=torch.float32))
        val_vecs.append(torch.tensor(va.values,  dtype=torch.float32))
        test_vecs.append(torch.tensor(te.values, dtype=torch.float32))

    def cat_ds(vecs):
        t = torch.cat(vecs)
        return TensorDataset(t)

    return cat_ds(train_vecs), cat_ds(val_vecs), cat_ds(test_vecs)


# ════════════════════════════════════════════════════════════════════════════════
# Anomaly threshold
# ════════════════════════════════════════════════════════════════════════════════

def compute_threshold(model: nn.Module, loader: DataLoader) -> float:
    """95th percentile of per-sample reconstruction MSE on training data."""
    model.eval()
    errors = []
    with torch.no_grad():
        for (x_b,) in loader:
            x_hat, _, _ = model(x_b)
            mse = ((x_hat - x_b) ** 2).mean(dim=1)   # per-sample MSE
            errors.append(mse.cpu())
    all_errors = torch.cat(errors).numpy()
    return float(np.percentile(all_errors, ANOMALY_PERCENTILE))


# ════════════════════════════════════════════════════════════════════════════════
# Latent scatter
# ════════════════════════════════════════════════════════════════════════════════

def save_latent_scatter(model: nn.Module, loader: DataLoader) -> None:
    """PCA-project latent μ to 2D and scatter-plot (colour by reconstruction error)."""
    model.eval()
    mus, errors = [], []
    with torch.no_grad():
        for (x_b,) in loader:
            x_hat, mu, _ = model(x_b)
            mse = ((x_hat - x_b) ** 2).mean(dim=1)
            mus.append(mu.cpu())
            errors.append(mse.cpu())

    mus    = torch.cat(mus).numpy()       # (N, latent_dim)
    errors = torch.cat(errors).numpy()    # (N,)

    # PCA to 2D — manual (no sklearn needed).
    centered = mus - mus.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T            # (N, 2)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=errors, cmap="plasma",
                     s=3, alpha=0.5, rasterized=True)
    plt.colorbar(sc, label="Recon MSE")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("WeatherVAE — Latent Space (test, PCA 2D)")
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=100)
    plt.close()
    print(f"  Latent scatter saved → {SCATTER_PATH}")


# ════════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════════

def train() -> None:
    print("Loading data ...")
    train_ds, val_ds, test_ds = load_data()
    print(
        f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} vectors"
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model    = WeatherVAE()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  WeatherVAE params: {n_params:,}")

    optimiser    = torch.optim.Adam(model.parameters(), lr=LR)
    train_recons, train_kls = [], []
    val_recons   = []

    print(
        f"\nTraining for up to {MAX_EPOCHS} epochs"
        f" (β 0→{BETA_MAX} over {BETA_ANNEAL} epochs) ..."
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        beta = _beta(epoch)

        # ── train ──
        model.train()
        t_recon = t_kl = 0.0
        for (x_b,) in train_dl:
            optimiser.zero_grad()
            x_hat, mu, logvar = model(x_b)
            loss, recon, kl   = elbo_loss(x_b, x_hat, mu, logvar, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_recon += recon.item() * len(x_b)
            t_kl    += kl.item()    * len(x_b)
        train_recon = t_recon / len(train_ds)
        train_kl    = t_kl    / len(train_ds)

        # ── val ──
        model.eval()
        v_recon = 0.0
        with torch.no_grad():
            for (x_b,) in val_dl:
                x_hat, mu, logvar = model(x_b)
                _, recon, _       = elbo_loss(x_b, x_hat, mu, logvar, beta)
                v_recon += recon.item() * len(x_b)
        val_recon = v_recon / len(val_ds)

        train_recons.append(train_recon)
        train_kls.append(train_kl)
        val_recons.append(val_recon)

        print(
            f"  epoch {epoch:02d}/{MAX_EPOCHS}  β={beta:.3f}"
            f"  train_recon={train_recon:.4f}  kl={train_kl:.4f}"
            f"  val_recon={val_recon:.4f}"
        )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\n  Weights saved → {WEIGHTS_PATH}")

    # ── training curve (dual axis: recon + KL) ──
    epochs = range(1, len(train_recons) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_recons, label="train recon", color="tab:blue")
    ax1.plot(epochs, val_recons,   label="val recon",   color="tab:cyan",  linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Recon MSE", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_kls, label="KL", color="tab:orange", linestyle=":")
    ax2.set_ylabel("KL divergence", color="tab:orange")
    ax1.set_title("WeatherVAE — Training Curve")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=100)
    plt.close(fig)
    print(f"  Training curve saved → {CURVE_PATH}")

    # ── anomaly threshold ──
    threshold = compute_threshold(model, train_dl)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold, "percentile": ANOMALY_PERCENTILE}, indent=2))
    print(f"  Anomaly threshold (p{ANOMALY_PERCENTILE}): {threshold:.6f}  → {THRESHOLD_PATH}")

    # ── test evaluation ──
    model.eval()
    test_errors = []
    with torch.no_grad():
        for (x_b,) in test_dl:
            x_hat, _, _ = model(x_b)
            mse = ((x_hat - x_b) ** 2).mean(dim=1)
            test_errors.append(mse.cpu())
    test_errors = torch.cat(test_errors).numpy()
    test_recon  = float(test_errors.mean())
    n_anomalies = int((test_errors > threshold).sum())
    anomaly_pct = 100 * n_anomalies / len(test_errors)

    print(f"\n  Test recon MSE: {test_recon:.4f}")
    print(f"  Anomalies in 2024 H2 test: {n_anomalies:,} / {len(test_errors):,} ({anomaly_pct:.1f}%)")

    # ── latent scatter ──
    save_latent_scatter(model, test_dl)

    print("\n  [grader] Copy test_recon + anomaly count into ml/README.md VAE row.")


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
