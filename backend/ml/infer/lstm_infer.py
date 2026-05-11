"""
lstm_infer.py — async inference wrapper for WeatherLSTM (baseline, not agent-wired).

Same interface as transformer_infer so the two can be called side-by-side for comparison.
"""

import json
from pathlib import Path
from typing import Optional

import torch

ROOT        = Path(__file__).parent.parent
MODEL_PATH  = ROOT / "models" / "lstm" / "model.pt"
SCALER_PATH = ROOT / "datasets" / "scaler_stats.json"

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

    from ml.train.train_lstm import WeatherLSTM
    m = WeatherLSTM()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _model = m

    with open(SCALER_PATH) as f:
        _stats = json.load(f)
    return True


async def lstm_forecast(city: str, hours: int = 24) -> dict:
    """LSTM baseline forecast — same shape as transformer_forecast for comparison."""
    if not _load_model():
        return {"error": "model not loaded", "detail": f"{MODEL_PATH} not found — run train_lstm.py"}

    try:
        import numpy as np
        import tools as wx_tools

        raw      = await wx_tools.get_hourly_forecast(city, INPUT_LEN)
        col_keys = ["temperature_2m", "relative_humidity_2m", "precipitation",
                    "wind_speed_10m", "weather_code", "surface_pressure", "cloud_cover"]
        rows = []
        for i in range(INPUT_LEN):
            rows.append([
                (raw.get(k, [0.0] * INPUT_LEN)[i] or 0.0) for k in col_keys
            ])

        arr = np.array(rows, dtype=np.float32)
        for j, feat in enumerate(FEATURES):
            if feat in _stats:
                mean = _stats[feat]["mean"]
                std  = _stats[feat]["std"] or 1.0
                arr[:, j] = (arr[:, j] - mean) / std

        x = torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            pred = _model(x).squeeze(0).tolist()

        t_mean = _stats["temperature_2m"]["mean"]
        t_std  = _stats["temperature_2m"]["std"] or 1.0
        pred_celsius = [round(p * t_std + t_mean, 1) for p in pred]

        return {
            "city":            city,
            "horizon_hours":   hours,
            "predicted_temps": pred_celsius[:hours],
            "model":           "lstm",
        }

    except Exception as e:
        return {"error": f"lstm inference failed: {str(e)[:200]}"}
