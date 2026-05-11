"""
rag_infer.py — FAISS-backed RAG over weather event snippets.

Index is built once at server startup via build_index() called from lifespan().
Per-request: encode query → L2-normalise → inner-product search → top-k snippets.

Model: sentence-transformers/all-MiniLM-L6-v2 (22M params, CPU-fast)
Index: faiss.IndexFlatIP  (inner product = cosine after L2-normalisation)
"""

from pathlib import Path
from typing import Optional

ROOT         = Path(__file__).parent.parent
SNIPPETS_PATH = ROOT / "datasets" / "weather_snippets.json"

# ── module-level cache ────────────────────────────────────────────────────────
_index    = None
_snippets: Optional[list[str]] = None
_embedder = None


def build_index() -> bool:
    """
    Load snippets + build FAISS index. Call from lifespan() at server startup.
    Returns True on success, False if snippets file is missing or deps unavailable.
    """
    global _index, _snippets, _embedder

    if not SNIPPETS_PATH.exists():
        print(f"[rag] snippets not found at {SNIPPETS_PATH} — RAG disabled")
        return False

    try:
        import json
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        with open(SNIPPETS_PATH, encoding="utf-8") as f:
            data = json.load(f)

        # Accept list of strings or list of {"text": ...} dicts
        _snippets = [
            s if isinstance(s, str) else s.get("text", str(s))
            for s in data
        ]

        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vecs = _embedder.encode(_snippets, convert_to_numpy=True, normalize_embeddings=True)
        vecs = vecs.astype(np.float32)

        _index = faiss.IndexFlatIP(vecs.shape[1])
        _index.add(vecs)
        print(f"[rag] index built: {len(_snippets)} snippets, dim={vecs.shape[1]}")
        return True

    except Exception as e:
        print(f"[rag] index build failed: {e}")
        return False


async def retrieve_weather_context(query: str, k: int = 3) -> dict:
    """Return top-k weather snippets most relevant to query."""
    if _index is None or _snippets is None or _embedder is None:
        if not SNIPPETS_PATH.exists():
            return {"error": "model not loaded", "detail": f"{SNIPPETS_PATH} not found"}
        # Try lazy build on first call (dev mode — lifespan may not have run)
        if not build_index():
            return {"error": "RAG index unavailable"}

    try:
        import numpy as np

        q_vec = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q_vec = q_vec.astype(np.float32)

        scores, indices = _index.search(q_vec, min(k, len(_snippets)))
        results = [
            {"snippet": _snippets[idx], "score": round(float(scores[0][i]), 4)}
            for i, idx in enumerate(indices[0])
            if idx >= 0
        ]
        return {"query": query, "results": results}

    except Exception as e:
        return {"error": f"RAG retrieval failed: {str(e)[:200]}"}
