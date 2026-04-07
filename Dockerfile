# =============================================================
# Dockerfile — Traffic Signal Optimization Environment
# FastAPI + Uvicorn on Python 3.11-slim
# =============================================================

# ── Stage 1: dependency builder ───────────────────────────────
FROM python:3.11-slim AS builder

# Install build tools needed for some packages (numpy, pydantic-core)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy fixed requirements first (layer-cache friendly)
COPY requirements.txt .

# Install into a prefix so we can copy only the site-packages
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ─────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY environment/ ./environment/
COPY tasks/       ./tasks/
COPY app.py       .

# .env is NOT copied — supply secrets via environment variables at runtime
# (see docker run -e or docker-compose env_file)

# Ensure Python finds local packages
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Port the FastAPI app listens on (matches app.py uvicorn.run port)
EXPOSE 7860

# Switch to non-root
USER appuser

# ── Health check ───────────────────────────────────────────────
# Hits the root endpoint; allows 30s startup time
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# ── Entrypoint ─────────────────────────────────────────────────
# Single-container / dev: uvicorn directly
# For production multi-worker, override CMD with:
#   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:7860
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
