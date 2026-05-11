import os
import json
import asyncio
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from groq import AsyncGroq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

import cache
import agent as ai_agent
import tools as wx_tools
from tts import synthesize as tts_synthesize

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client: Optional[AsyncGroq] = None
_STARTUP_TIME = _time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client
    cache.init_cache(None)
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    for name, mod_path, fn in [
        ("transformer", "ml.infer.transformer_infer", "_load_model"),
        ("lstm",        "ml.infer.lstm_infer",        "_load_model"),
        ("vae",         "ml.infer.vae_infer",         "_load_model"),
        ("rag",         "ml.infer.rag_infer",         "build_index"),
        ("clip",        "ml.infer.clip_infer",        "_load_model"),
        ("cnn",         "ml.infer.cnn_infer",         "_load_model"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            ok = getattr(mod, fn)()
            print(f"[ml] {name}: {'loaded' if ok else 'SKIPPED (weights missing)'}")
        except Exception as e:
            print(f"[ml] {name}: SKIPPED ({e})")

    yield


app = FastAPI(title="Weathering With You", lifespan=lifespan)


# ── WebSocket chat ─────────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    message_history: list[dict] = []

    await ws.send_json({"type": "bot_state", "state": "SCANNING", "temperature": 25})
    await asyncio.sleep(1.2)
    await ws.send_json({"type": "bot_state", "state": "IDLE", "temperature": 25})

    try:
        while True:
            data = await ws.receive_json()
            user_msg = data.get("message", "")
            if not user_msg.strip():
                continue

            await ws.send_json({"type": "thinking", "state": "SPEAKING"})

            try:
                text, zero_state = await ai_agent.run_agent(
                    user_message=user_msg,
                    message_history=message_history,
                    groq_client=groq_client,
                )
                message_history.append({"role": "user", "content": user_msg})
                message_history.append({"role": "assistant", "content": text})
                if len(message_history) > 20:
                    message_history = message_history[-20:]

                await ws.send_json({
                    "type": "response",
                    "text": text,
                    "bot_state": zero_state,
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                await ws.send_json({
                    "type": "error",
                    "text": f"yaar... kuch toh gadbad ho gayi. ({str(e)[:120]})",
                    "bot_state": {"state": "IDLE", "temperature": 25},
                })

    except WebSocketDisconnect:
        pass


# ── Weather endpoints ──────────────────────────────────────────────────────────
@app.get("/api/weather/current")
async def current_weather(city: str = "Lahore", lat: float = None, lon: float = None, timezone: str = None):
    try:
        return await wx_tools.get_current_weather(city, lat, lon, timezone)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/weather/hourly")
async def hourly_forecast(city: str = "Lahore", hours: int = 24):
    try:
        return await wx_tools.get_hourly_forecast(city, hours)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/weather/7day")
async def forecast_7day(city: str = "Lahore"):
    try:
        return await wx_tools.get_7day_forecast(city)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/weather/minutely")
async def minutely_rain(lat: float, lon: float):
    try:
        return await wx_tools.get_minutely_rain(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/weather/aqi")
async def air_quality(lat: float, lon: float):
    try:
        return await wx_tools.get_air_quality(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/activity")
async def activity_index(city: str = "Lahore", activity: str = "cricket"):
    try:
        return await wx_tools.get_activity_index(city, activity)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    try:
        audio = await tts_synthesize(req.text)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)[:200]}")


@app.get("/api/search")
async def search_city(q: str):
    try:
        return await wx_tools.search_city(q)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Live ML forecast ───────────────────────────────────────────────────────────
@app.get("/api/ml/forecast")
async def ml_forecast(city: str = "Lahore", hours: int = 24):
    """
    Live 24-hour temperature forecast from the champion Transformer model,
    plus VAE anomaly detection on current conditions.

    Example: GET /api/ml/forecast?city=Karachi&hours=24
    """
    import datetime as dt
    from ml.infer import transformer_infer, vae_infer

    # Run forecast + anomaly detection concurrently
    forecast, anomaly = await asyncio.gather(
        transformer_infer.transformer_forecast(city, hours),
        vae_infer.detect_weather_anomaly(city),
    )

    # Surface model-not-loaded as a clean 503 instead of a 200 with error body
    if "error" in forecast:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Transformer not loaded: {forecast['error']}. "
                "Run train_transformer.py to generate weights."
            ),
        )

    return {
        "city": city,
        "model": "transformer",
        "champion_run": "txf-003",
        "val_mse": 0.2543,
        "horizon_hours": hours,
        "forecast_celsius": forecast.get("predicted_temps", []),
        "anomaly": (
            {
                "is_anomaly": anomaly.get("anomaly", False),
                "score": anomaly.get("score"),
                "threshold": anomaly.get("threshold"),
                "model": "vae",
                "champion_run": "vae-003",
            }
            if "error" not in anomaly
            else {"error": anomaly["error"]}
        ),
        "generated_at": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── MLOps ──────────────────────────────────────────────────────────────────────
ML_DIR = Path(__file__).parent / "ml"


@app.get("/api/mlops/data")
async def mlops_data():
    """Aggregated MLOps data for the dashboard -- reads all JSON artifacts."""
    try:
        result = {}
        for key, rel in [
            ("runs",         "experiments/runs.json"),
            ("best_metrics", "experiments/best_metrics.json"),
            ("manifest",     "datasets/manifest.json"),
            ("comparison",   "experiments/comparison_table.json"),
            ("retrain",      "experiments/retrain_log.json"),
        ]:
            p = ML_DIR / rel
            if p.exists():
                result[key] = json.loads(p.read_text())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlops")
async def mlops_dashboard():
    dashboard = ML_DIR / "dashboard.html"
    if dashboard.exists():
        return FileResponse(str(dashboard), media_type="text/html")
    raise HTTPException(status_code=404, detail="Dashboard not found.")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """
    System health endpoint. Returns model load status, MLOps artifact
    presence, Redis status, and last retrain info.
    """
    # Model status
    model_status = {}
    for name, mod_path, attr in [
        ("transformer", "ml.infer.transformer_infer", "_model"),
        ("lstm",        "ml.infer.lstm_infer",        "_model"),
        ("cnn",         "ml.infer.cnn_infer",         "_model"),
        ("vae",         "ml.infer.vae_infer",         "_model"),
        ("rag",         "ml.infer.rag_infer",         "_index"),
        ("clip",        "ml.infer.clip_infer",        "_model"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            loaded = getattr(mod, attr, None) is not None
            model_status[name] = "loaded" if loaded else "not_loaded"
        except Exception:
            model_status[name] = "unavailable"

    # MLOps artifacts
    artifacts = {}
    for key, rel in [
        ("config",       "ml/config.yaml"),
        ("manifest",     "ml/datasets/manifest.json"),
        ("runs",         "ml/experiments/runs.json"),
        ("best_metrics", "ml/experiments/best_metrics.json"),
        ("comparison",   "ml/experiments/comparison_table.json"),
        ("retrain_log",  "ml/experiments/retrain_log.json"),
        ("dashboard",    "ml/dashboard.html"),
    ]:
        p = Path(__file__).parent / rel
        artifacts[key] = {
            "exists": p.exists(),
            "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else 0,
        }

    # Redis
    redis_ok = False
    try:
        if cache._redis and cache._redis.ping():
            redis_ok = True
    except Exception:
        pass

    # Last retrain
    retrain_info = {"last_retrain": "never", "trigger": "N/A"}
    retrain_path = ML_DIR / "experiments" / "retrain_log.json"
    if retrain_path.exists():
        try:
            log = json.loads(retrain_path.read_text())
            last = log.get("entries", [{}])[-1]
            retrain_info = {
                "last_retrain":   last.get("timestamp", "never"),
                "trigger":        last.get("trigger", "N/A"),
                "champion_model": last.get("champion_model", "N/A"),
                "val_mse":        last.get("val_mse", "N/A"),
                "next_scheduled": log.get("next_scheduled", "N/A"),
            }
        except Exception:
            pass

    uptime_s = round(_time.time() - _STARTUP_TIME)

    return {
        "status":       "ok",
        "uptime_s":     uptime_s,
        "uptime_human": f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m {uptime_s % 60}s",
        "models":       model_status,
        "artifacts":    artifacts,
        "redis":        "connected" if redis_ok else "unavailable",
        "retrain":      retrain_info,
        "api_version":  "1.0.0",
    }


# ── Serve React SPA in production ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
