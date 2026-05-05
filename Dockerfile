# ──────────────────────────────────────────────────────────────────
# StackMind — Production Backend Dockerfile
# Multi-stage build for a lean, secure production image.
# ──────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install gunicorn

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="StackMind Team"
LABEL description="StackMind — Multi-step reasoning AI agent"
LABEL version="1.0.0"

# Runtime system dependencies (OpenCV, Tesseract, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r stackmind && useradd -r -g stackmind -d /app -s /sbin/nologin stackmind

WORKDIR /app

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/memory logs && \
    chown -R stackmind:stackmind /app

# Switch to non-root user
USER stackmind

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=info

# Expose API port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Start with Gunicorn + Uvicorn workers
CMD gunicorn interfaces.api:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${WORKERS} \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL}
