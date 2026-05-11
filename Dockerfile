# ── Weathering With You — HuggingFace Spaces Dockerfile ──────────────────────
# Runs the FastAPI backend on port 7860 (HF Spaces default).
# Static React build is served by FastAPI itself (STATIC_DIR in main.py).

FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user
RUN useradd -m -u 1000 hfuser

WORKDIR /app

# System deps (audio, numpy, faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling 2 GB CUDA wheels)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.2.0+cpu torchvision==0.17.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY frontend/dist/ ./frontend/dist/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app/backend

EXPOSE 7860

USER hfuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
