"""
data_pipeline.py — fetch, clean, standardise, and window weather data.

Usage:
    python -m ml.data_pipeline            # skip cities whose parquet already exists
    python -m ml.data_pipeline --force    # re-download everything

Outputs (all under backend/ml/datasets/):
    <city>.parquet          one file per city, full 2023-2024 hourly record
    scaler_stats.json       train-split mean/std per feature (no val/test leakage)
    imputation_log.json     any feature with NaN rate >1% after forward-fill

Time split (NEVER random-split):
    train : 2023-01-01 – 2023-12-31
    val   : 2024-01-01 – 2024-06-30
    test  : 2024-07-01 – 2024-12-31
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch

# ── reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── paths ────────────────────────────────────────────────────────────────────
DATASETS_DIR = Path(__file__).parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH     = DATASETS_DIR / "scaler_stats.json"
IMPUTATION_PATH = DATASETS_DIR / "imputation_log.json"

# ── cities (mirrors tools.PK_CITIES) ─────────────────────────────────────────
PK_CITIES = {
    "karachi":   {"lat": 24.8607, "lon": 67.0011},
    "lahore":    {"lat": 31.5497, "lon": 74.3436},
    "islamabad": {"lat": 33.6844, "lon": 73.0479},
    "topi":      {"lat": 34.0700, "lon": 72.6200},
    "peshawar":  {"lat": 34.0150, "lon": 71.5249},
    "quetta":    {"lat": 30.1798, "lon": 66.9750},
}

# ── feature set ──────────────────────────────────────────────────────────────
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
    "surface_pressure",
    "cloud_cover",
]

# ── time ranges ───────────────────────────────────────────────────────────────
DATE_START = "2023-01-01"
DATE_END   = "2024-12-31"

TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END   = "2024-06-30"
TEST_START = "2024-07-01"

# ── Open-Meteo archive endpoint (NOT /forecast — only goes ~92 days back) ─────
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MAX_FILL_HOURS = 3          # forward-fill at most 3 consecutive NaN hours
NAN_WARN_PCT   = 0.01       # log to imputation_log if NaN rate exceeds 1% after fill


# ── fetch ─────────────────────────────────────────────────────────────────────

async def fetch_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    """Download hourly data for one city from Open-Meteo Archive API."""
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(FEATURES),
        "start_date":      DATE_START,
        "end_date":        DATE_END,
        "timezone":        "Asia/Karachi",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(ARCHIVE_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df["city"] = city
    return df


# ── clean ─────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, city: str, imputation_log: dict) -> pd.DataFrame:
    """Forward-fill NaNs (≤3h gap); log features that exceed 1% NaN after fill."""
    for feat in FEATURES:
        if feat not in df.columns:
            continue

        pre_nan = df[feat].isna().sum()
        if pre_nan == 0:
            continue

        # Forward-fill limited to MAX_FILL_HOURS consecutive gaps.
        df[feat] = df[feat].ffill(limit=MAX_FILL_HOURS)

        post_nan = df[feat].isna().sum()
        nan_rate = post_nan / len(df)

        if nan_rate > NAN_WARN_PCT:
            key = f"{city}.{feat}"
            imputation_log[key] = {
                "pre_fill_nans":  int(pre_nan),
                "post_fill_nans": int(post_nan),
                "nan_rate_pct":   round(nan_rate * 100, 3),
            }
            print(
                f"  [warn] {city}/{feat}: {nan_rate*100:.2f}% NaN after fill "
                f"({post_nan} rows)"
            )

    return df


# ── standardise ───────────────────────────────────────────────────────────────

def compute_scaler_stats(all_cities: dict[str, pd.DataFrame]) -> dict:
    """
    Compute mean/std from the TRAIN split only (2023).
    Never touch val/test rows — no leakage.
    """
    train_frames = [
        df.loc[df.index <= TRAIN_END, FEATURES]
        for df in all_cities.values()
    ]
    train_all = pd.concat(train_frames, axis=0)

    stats = {}
    for feat in FEATURES:
        col = train_all[feat].dropna()
        stats[feat] = {"mean": float(col.mean()), "std": float(col.std())}
    return stats


def apply_scaler(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Z-score each feature using train-derived mean/std."""
    df = df.copy()
    for feat in FEATURES:
        if feat in stats:
            mean = stats[feat]["mean"]
            std  = stats[feat]["std"] or 1.0   # guard zero-std features
            df[feat] = (df[feat] - mean) / std
    return df


# ── sliding window ────────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    input_len: int = 48,
    horizon: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slide a window over df and return (X, y) tensors.

    X shape: (N, input_len, num_features)
    y shape: (N, horizon)              — temperature_2m only

    Only uses FEATURES columns; drops rows with any NaN in the window.
    """
    arr = df[FEATURES].values.astype(np.float32)   # (T, F)
    temp_idx = FEATURES.index("temperature_2m")

    xs, ys = [], []
    total = len(arr)
    for i in range(total - input_len - horizon + 1):
        window = arr[i : i + input_len]
        target = arr[i + input_len : i + input_len + horizon, temp_idx]

        if np.isnan(window).any() or np.isnan(target).any():
            continue

        xs.append(window)
        ys.append(target)

    if not xs:
        return torch.empty(0), torch.empty(0)

    X = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    return X, y


# ── split helper ──────────────────────────────────────────────────────────────

def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) slices. Index must be a DatetimeIndex."""
    train = df.loc[df.index <= TRAIN_END]
    val   = df.loc[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df.loc[df.index >= TEST_START]
    return train, val, test


# ── main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(force: bool = False) -> dict[str, pd.DataFrame]:
    imputation_log: dict = {}
    all_dfs: dict[str, pd.DataFrame] = {}

    print(f"Pipeline start — {len(PK_CITIES)} cities, {DATE_START} → {DATE_END}")
    print(f"Features: {FEATURES}\n")

    for city, coords in PK_CITIES.items():
        parquet_path = DATASETS_DIR / f"{city}.parquet"

        if parquet_path.exists() and not force:
            print(f"[skip] {city}: parquet exists — use --force to re-download")
            df = pd.read_parquet(parquet_path)
            df.index = pd.to_datetime(df.index)
            all_dfs[city] = df
            train, val, test = split_df(df)
            print(
                f"       rows total={len(df):,}  "
                f"train={len(train):,}  val={len(val):,}  test={len(test):,}"
            )
            continue

        print(f"[fetch] {city} ({coords['lat']}, {coords['lon']}) ...")
        df = await fetch_city(city, **coords)
        df = clean(df, city, imputation_log)

        train, val, test = split_df(df)
        print(
            f"        rows total={len(df):,}  "
            f"train={len(train):,}  val={len(val):,}  test={len(test):,}"
        )

        # Show per-feature NaN summary after cleaning.
        nan_counts = df[FEATURES].isna().sum()
        non_zero = nan_counts[nan_counts > 0]
        if non_zero.empty:
            print(f"        NaN remaining: none ✓")
        else:
            for feat, cnt in non_zero.items():
                print(f"        NaN remaining: {feat}={cnt}")

        df.to_parquet(parquet_path)
        print(f"        saved → {parquet_path}\n")
        all_dfs[city] = df

    # Scaler stats from train split only.
    print("Computing scaler stats (train split only — no leakage) ...")
    stats = compute_scaler_stats(all_dfs)
    SCALER_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Saved → {SCALER_PATH}")
    for feat, s in stats.items():
        print(f"  {feat:30s}  mean={s['mean']:8.3f}  std={s['std']:7.3f}")

    # Save imputation log (even if empty).
    IMPUTATION_PATH.write_text(json.dumps(imputation_log, indent=2))
    if imputation_log:
        print(f"\nImputation warnings written → {IMPUTATION_PATH}")
    else:
        print(f"\nNo imputation warnings (all features <1% NaN) → {IMPUTATION_PATH}")

    print("\nPipeline complete.")
    return all_dfs


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and preprocess PK weather data.")
    parser.add_argument("--force", action="store_true", help="Re-download even if parquet exists.")
    args = parser.parse_args()

    asyncio.run(run_pipeline(force=args.force))
