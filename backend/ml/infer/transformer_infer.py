"""
transformer_infer.py — async inference wrapper for WeatherTransformer.

Lazy-loads weights on first call. Missing weights → graceful error dict.
Called by agent tool: transformer_forecast(city, hours=24)
"""

import json
from pathlib import Path
from typing import Optional

import torch

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "transformer" / "model.pt"
SCALER_PATH = ROOT / "datasets" / "scaler_stats.json"

# ── module-level cache ────────────────────────────────────────────────────────
_model = None
_stats: Optional[dict] = None

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "weather_code", "surface_pressure", "cloud_cover",
]
INPUT_LEN = 48


def _load_model():
    global _model, _stats
    if _model is not None:
        return True

    if not MODEL_PATH.exists():
        return False

    # Import here so missing torch doesn't break server startup
    from ml.train.train_transformer import WeatherTransformer
    m = WeatherTransformer()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _model = m

    with open(SCALER_PATH) as f:
        _stats = json.load(f)
    return True


async def transformer_forecast(city: str, hours: int = 24) -> dict:
    """
    Fetch last 48h of data for city, run WeatherTransformer, return 24h forecast.
    """
    if not _load_model():
        return {"error": "model not loaded", "detail": f"{MODEL_PATH} not found — run train_transformer.py"}

    try:
        import numpy as np
        import tools as wx_tools

        # Fetch recent 48h from Open-Meteo live
        raw = await wx_tools.get_hourly_forecast(city, INPUT_LEN)
        temps   = raw.get("temperature_2m",       [None] * INPUT_LEN)
        humidity = raw.get("relative_humidity_2m", [None] * INPUT_LEN)
        precip  = raw.get("precipitation",         [None] * INPUT_LEN)
        wind    = raw.get("wind_speed_10m",        [None] * INPUT_LEN)
        wcode   = raw.get("weather_code",          [None] * INPUT_LEN)
        pressure = raw.get("surface_pressure",     [None] * INPUT_LEN)
        clouds  = raw.get("cloud_cover",           [None] * INPUT_LEN)

        # Assemble feature matrix (48, 7)
        rows = []
        for i in range(INPUT_LEN):
            row = [temps[i], humidity[i], precip[i], wind[i],
                   wcode[i], pressure[i], clouds[i]]
            # Forward-fill any None with 0 (standardised space — neutral)
            rows.append([v if v is not None else 0.0 for v in row])

        arr = np.array(rows, dtype=np.float32)

        # Apply train-scaler
        for j, feat in enumerate(FEATURES):
            if feat in _stats:
                mean = _stats[feat]["mean"]
                std  = _stats[feat]["std"] or 1.0
                arr[:, j] = (arr[:, j] - mean) / std

        x = torch.tensor(arr).unsqueeze(0)   # (1, 48, 7)
        with torch.no_grad():
            pred = _model(x).squeeze(0).tolist()   # (24,)

        # De-standardise temperature predictions
        t_mean = _stats["temperature_2m"]["mean"]
        t_std  = _stats["temperature_2m"]["std"] or 1.0
        pred_celsius = [round(p * t_std + t_mean, 1) for p in pred]

        return {
            "city":            city,
            "horizon_hours":   hours,
            "predicted_temps": pred_celsius[:hours],
            "model":           "transformer",
        }

    except Exception as e:
        return {"error": f"transformer inference failed: {str(e)[:200]}"}
