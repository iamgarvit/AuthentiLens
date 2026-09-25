# AuthentiLens demo - Hugging Face Docker Space (CPU only).
#
# Build context is the directory assembled by hf_space/sync.py, which places the
# repository's app/ and authentilens/ packages next to this file so the Space
# runs the same code as the repo.
FROM python:3.11-slim

# OpenCV (used by the optional Noiseprint analysis) needs these at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; everything the app writes must be
# writable by that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Hugging Face Hub cache and the downloaded checkpoints, both under $HOME.
ENV HF_HOME=/home/user/.cache/huggingface \
    AUTHENTILENS_CHECKPOINT_DIR=/home/user/checkpoints

WORKDIR /home/user/app

# CPU-only torch: the default wheels bundle CUDA and are several GB.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY --chown=user . .

# Must match app_port in README.md and the port in .streamlit/config.toml.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
