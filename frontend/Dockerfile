# ============================================================================
# ApertureAuthentiLens — Frontend Dockerfile
# ============================================================================
# Python 3.10 slim image with Streamlit, Requests, Matplotlib, and OpenCV.
# ============================================================================

FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Launch Streamlit with server settings for Docker
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
