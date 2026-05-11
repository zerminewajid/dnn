"""
vae_infer.py — async inference wrapper for WeatherVAE anomaly detector.

Fetches current weather for a city, standardises, runs VAE, compares
reconstruction MSE against the pre-computed 95th-percentile threshold.
"""

import json
from pathlib import Path
from typing import Optional

import torch

ROOT            = Path(__file__).parent.parent
MODEL_PATH      = ROOT / "models" / "vae" / "model.pt"
THRESHOLD_PATH  = ROOT / "models" / "vae" / "anomaly_threshold.json"
SCALER_PATH     = ROOT / "datasets" / "scaler_stats.json"

_model = None
_stats: Optional[dict] = None
_threshold: Optional[float] = None

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "weather_code", "surface_pressure", "cloud_cover",
]


def _load_model():
    global _model, _stats, _threshold
    if _model is not None:
        return True

    if not MODEL_PATH.exists():
        return False

    from ml.train.train_vae import WeatherVAE
    m = WeatherVAE()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _model = m

    with open(SCALER_PATH) as f:
        _stats = json.load(f)

    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            _threshold = json.load(f)["threshold"]
    else:
        _threshold = 1.0   # fallback — will flag almost nothing
    return True


async def detect_weather_anomaly(city: str) -> dict:
    """
    Fetch current weather for city, run VAE, return anomaly flag + score.
    """
    if not _load_model():
        return {"error": "model not loaded", "detail": f"{MODEL_PATH} not found — run train_vae.py"}

    try:
        import numpy as np
        import tools as wx_tools

        current = await wx_tools.get_current_weather(city)

        # Build single feature vector
        vec = np.array([
            current.get("temperature_2m",       0.0) or 0.0,
            current.get("relative_humidity_2m", 0.0) or 0.0,
            current.get("precipitation",        0.0) or 0.0,
            current.get("wind_speed_10m",       0.0) or 0.0,
            current.get("weather_code",         0.0) or 0.0,
            current.get("surface_pressure",     0.0) or 0.0,
            current.get("cloud_cover",          0.0) or 0.0,
        ], dtype=np.float32)

        # Standardise with train stats
        for j, feat in enumerate(FEATURES):
            if feat in _stats:
                mean = _stats[feat]["mean"]
                std  = _stats[feat]["std"] or 1.0
                vec[j] = (vec[j] - mean) / std

        x = torch.tensor(vec).unsqueeze(0)   # (1, 7)
        with torch.no_grad():
            x_hat, _, _ = _model(x)
        score = float(((x_hat - x) ** 2).mean().item())

        return {
            "city":      city,
            "anomaly":   score > _threshold,
            "score":     round(score, 6),
            "threshold": round(_threshold, 6),
        }

    except Exception as e:
        return {"error": f"VAE inference failed: {str(e)[:200]}"}
