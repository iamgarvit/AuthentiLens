# ============================================================================
# AuthentiLens — Backend Dockerfile
# ============================================================================
# Python 3.10 slim image with PyTorch, segmentation-models-pytorch, FastAPI,
# Uvicorn and OpenCV.
#
# Build context is the REPO ROOT (see docker-compose.yml) so the shared
# authentilens/ package can be copied in. Checkpoints are NOT baked into the
# image; docker-compose mounts ./checkpoints at /app/checkpoints.
# ============================================================================

FROM python:3.10-slim

# System dependencies for OpenCV and curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY services/backend/requirements.txt services/backend/requirements.txt
RUN pip install --no-cache-dir -r services/backend/requirements.txt

# Copy application source code (same layout as the repo)
COPY authentilens/ authentilens/
COPY services/backend/ services/backend/

ENV PYTHONPATH=/app \
    AUTHENTILENS_CHECKPOINT_DIR=/app/checkpoints
WORKDIR /app/services/backend

# Expose the FastAPI server port
EXPOSE 8000

# Launch the FastAPI server via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
