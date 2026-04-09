# ── Dockerfile ────────────────────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean by separating dependency
# installation from the runtime layer.
#
# Build:  docker build -t delhi-aq-dashboard .
# Run:    docker run -p 7860:7860 delhi-aq-dashboard
# HF:     Push to HF Spaces (Dockerfile SDK) — port 7860 is required by HF.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — only reinstall if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ ./app/
COPY dataset/features/ ./dataset/features/

# HF Spaces expects the app to listen on port 7860
EXPOSE 7860

# Streamlit config for HF Spaces
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the dashboard
CMD ["streamlit", "run", "app/dashboard.py",
     "--server.port=7860",
     "--server.address=0.0.0.0",
     "--server.headless=true"]
