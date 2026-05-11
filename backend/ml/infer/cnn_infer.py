"""
cnn_infer.py — async inference wrapper for SkyCNN (not agent-wired; used for CLIP comparison).
"""

import io
from pathlib import Path
from typing import Optional, Union

import torch

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "cnn" / "model.pt"

CLASSES = ["clear", "cloudy", "rainy"]

_model = None


def _load_model():
    global _model
    if _model is not None:
        return True
    if not MODEL_PATH.exists():
        return False
    from ml.train.train_cnn import SkyCNN
    m = SkyCNN()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _model = m
    return True


def _preprocess(source: Union[str, bytes]):
    import torchvision.transforms as T
    from PIL import Image
    tf = T.Compose([
        T.Resize(144),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    else:
        import httpx
        resp = httpx.get(source, timeout=10.0)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return tf(img).unsqueeze(0)   # (1, 3, 128, 128)


async def classify_sky_cnn(image_source: Union[str, bytes]) -> dict:
    """3-class CNN sky classification (baseline vs CLIP zero-shot)."""
    if not _load_model():
        return {"error": "model not loaded", "detail": f"{MODEL_PATH} not found — run train_cnn.py"}

    try:
        import asyncio
        import torch.nn.functional as F

        def _infer():
            x = _preprocess(image_source)
            with torch.no_grad():
                logits = _model(x).squeeze(0)
                probs  = F.softmax(logits, dim=0).tolist()
            best   = int(max(range(len(probs)), key=lambda i: probs[i]))
            return CLASSES[best], round(probs[best], 4), {c: round(probs[i], 4) for i, c in enumerate(CLASSES)}

        label, confidence, all_probs = await asyncio.to_thread(_infer)
        return {"label": label, "confidence": confidence, "probabilities": all_probs}

    except Exception as e:
        return {"error": f"CNN inference failed: {str(e)[:200]}"}
