"""
clip_infer.py — zero-shot sky classification via openai/clip-vit-base-patch32.

6 CLIP labels → collapsed to 3 CNN-compatible labels via map_to_cnn_labels().
Sunset maps to "overcast" (conservative — Section 2 fix).
"""

import io
from pathlib import Path
from typing import Optional, Union

_model     = None
_processor = None

CLIP_LABELS = [
    "clear sky",
    "partly cloudy sky",
    "overcast sky",
    "rainy sky",
    "foggy sky",
    "sunset sky",
]

def map_to_cnn_labels(clip_label: str) -> str:
    """
    Collapse 6 CLIP labels → 3 CNN labels.
    sunset → overcast (conservative; CNN has no sunset class).
    foggy  → overcast (closest visual analogue).
    """
    mapping = {
        "clear sky":        "clear",
        "partly cloudy sky":"cloudy",
        "overcast sky":     "cloudy",
        "rainy sky":        "rainy",
        "foggy sky":        "cloudy",
        "sunset sky":       "cloudy",  # sunset-exclusion rule
    }
    return mapping.get(clip_label, "clear")


def _load_model():
    global _model, _processor
    if _model is not None:
        return True
    try:
        from transformers import CLIPProcessor, CLIPModel
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
        return True
    except Exception as e:
        print(f"[clip] model load failed: {e}")
        return False


def _load_image(source: Union[str, bytes]):
    """Accept URL string, base64 bytes, or raw image bytes."""
    from PIL import Image
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    # URL
    import httpx
    resp = httpx.get(source, timeout=10.0)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


async def classify_sky_image(image_source: Union[str, bytes]) -> dict:
    """
    Zero-shot classify a sky image.

    image_source: URL string or raw image bytes.
    Returns {"clip_label", "clip_confidence", "cnn_compatible_label"}.
    """
    if not _load_model():
        return {"error": "model not loaded", "detail": "CLIP failed to load — check transformers install"}

    try:
        import torch
        import asyncio

        def _infer():
            image = _load_image(image_source)
            inputs = _processor(
                text=CLIP_LABELS,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                outputs    = _model(**inputs)
                logits     = outputs.logits_per_image   # (1, n_labels)
                probs      = logits.softmax(dim=-1).squeeze(0).tolist()

            best_idx    = int(max(range(len(probs)), key=lambda i: probs[i]))
            clip_label  = CLIP_LABELS[best_idx]
            confidence  = round(probs[best_idx], 4)
            cnn_label   = map_to_cnn_labels(clip_label)
            return clip_label, confidence, cnn_label

        clip_label, confidence, cnn_label = await asyncio.to_thread(_infer)

        return {
            "clip_label":          clip_label,
            "clip_confidence":     confidence,
            "cnn_compatible_label": cnn_label,
        }

    except Exception as e:
        return {"error": f"CLIP inference failed: {str(e)[:200]}"}
