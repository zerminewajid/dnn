"""
generate_manifest.py — Regenerate datasets/manifest.json from scratch
======================================================================
Reads every .parquet file in ml/datasets/, computes real MD5 checksums
and byte sizes, pulls scaler stats from scaler_stats.json, and writes
a fresh manifest.json.

Usage (run from repo root with .venv active):
    python backend/ml/generate_manifest.py

Requirements: pyarrow or fastparquet (already in .venv for ML work)
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
DATASETS = HERE / "datasets"
CONFIG   = HERE / "config.yaml"

CITIES = {
    "islamabad": (33.6844, 73.0479),
    "karachi":   (24.8607, 67.0011),
    "lahore":    (31.5204, 74.3587),
    "peshawar":  (34.015,  71.5249),
    "quetta":    (30.1798, 66.975),
    "topi":      (34.074,  72.6145),
}

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "weather_code", "surface_pressure", "cloud_cover",
]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    """Count rows without loading full dataframe — uses parquet metadata."""
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(str(path)).num_rows
    except ImportError:
        pass
    try:
        import pandas as pd
        return len(pd.read_parquet(path, columns=["temperature_2m"]))
    except Exception:
        pass
    # fallback: assume 2 years of hourly data
    return 17520


def load_config_features():
    try:
        import yaml
        with open(CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg["shared"]["features"], cfg["shared"]["input_len"], cfg["shared"]["horizon"]
    except Exception:
        return FEATURES, 48, 24


def main():
    print("Weathering With You — Manifest Generator")
    print("=" * 50)

    features, input_len, horizon = load_config_features()

    # ── Scaler stats ──────────────────────────────────────────────────────────
    scaler_path = DATASETS / "scaler_stats.json"
    scaler_stats_raw = {}
    if scaler_path.exists():
        raw = json.loads(scaler_path.read_text())
        for feat, vals in raw.items():
            scaler_stats_raw[feat] = {
                "mean": round(vals["mean"], 3),
                "std":  round(vals["std"],  3),
            }
    else:
        print("  ⚠️  scaler_stats.json not found — using zeros")

    # ── Per-city stats ────────────────────────────────────────────────────────
    cities_out = {}
    total_bytes = 0

    for city, (lat, lon) in CITIES.items():
        pq_path = DATASETS / f"{city}.parquet"
        if not pq_path.exists():
            print(f"  ⚠️  {city}.parquet not found — skipping")
            continue

        size  = pq_path.stat().st_size
        cksum = md5_of(pq_path)
        rows  = count_rows(pq_path)
        total_bytes += size

        cities_out[city] = {
            "lat": lat, "lon": lon,
            "timezone": "Asia/Karachi",
            "file":  f"{city}.parquet",
            "bytes": size,
            "md5":   cksum,
            "rows":  rows,
            "nan_pct": 0.0,
        }
        print(f"  {city:12} {size/1024:.0f} KB  {rows} rows  md5={cksum[:12]}…")

    # ── Assemble manifest ──────────────────────────────────────────────────────
    manifest = {
        "_comment": "Auto-generated — run generate_manifest.py to refresh.",
        "schema_version":   "1.0",
        "generated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version": "1.2.0",
        "config_ref":       "../config.yaml",

        "date_range": {
            "start": "2023-01-01", "end": "2024-12-31",
            "total_hours": 17520, "frequency": "1h",
        },
        "splits": {
            "train": {"start": "2023-01-01", "end": "2023-12-31", "hours": 8760},
            "val":   {"start": "2024-01-01", "end": "2024-06-30", "hours": 4344},
            "test":  {"start": "2024-07-01", "end": "2024-12-31", "hours": 4416},
        },

        "features":   features,
        "target":     "temperature_2m",
        "input_len":  input_len,
        "horizon":    horizon,

        "scaler":       "standard",
        "scaler_stats": scaler_stats_raw,

        "imputation": {
            "strategy":       "forward_fill",
            "max_fill_hours": 3,
            "nan_warn_pct":   0.01,
        },

        "cities": cities_out,

        "totals": {
            "cities":      len(cities_out),
            "total_rows":  sum(c["rows"] for c in cities_out.values()),
            "total_bytes": total_bytes,
            "total_mb":    round(total_bytes / 1024 / 1024, 2),
        },

        "source":      "Open-Meteo Historical Weather API",
        "source_url":  "https://archive-api.open-meteo.com/v1/archive",
        "license":     "CC BY 4.0",
    }

    out_path = DATASETS / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))

    print()
    print(f"✅ manifest.json written → {out_path}")
    print(f"   Cities:     {manifest['totals']['cities']}")
    print(f"   Total rows: {manifest['totals']['total_rows']:,}")
    print(f"   Total size: {manifest['totals']['total_mb']} MB")
    print(f"   Generated:  {manifest['generated_at']}")


if __name__ == "__main__":
    main()
