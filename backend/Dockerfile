# ============================================================================
# ApertureAuthentiLens — Backend Dockerfile
# ============================================================================
# Python 3.10 slim image with PyTorch (CPU), FastAPI, Uvicorn, OpenCV,
# HuggingFace Transformers (for CLIP), and supporting libraries.
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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the FastAPI server port
EXPOSE 8000

# Launch the FastAPI server via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
