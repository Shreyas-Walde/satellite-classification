# ============================================
# Satellite Image Classifier - Docker Image
# Multi-stage build for minimal final image
# ============================================

# ---- Stage 1: Builder ----
# Install all build-time dependencies here so they don't
# bloat the final image (gcc, pip cache, wheel build artifacts).
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# gcc is needed to compile some Python C-extensions (e.g. uvloop)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (separate step → own Docker cache layer)
# Using the CPU wheel index avoids pulling ~2 GB of CUDA libraries
RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
# Start from a fresh slim image — no gcc, no pip cache, no build artifacts.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the minimal runtime library that OpenCV needs
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy the pre-built Python packages from the builder stage
COPY --from=builder /install /usr/local

# ---- Application code ----
# Copy only the files the API actually imports at runtime.
# Anything not listed here (notebooks, scripts, data, training code)
# stays out of the image → smaller + more secure.
COPY main.py .
COPY model.py .
COPY preprocessing.py .

# ---- Model weights ----
# Separate layer so code changes don't re-copy the (large) model files.
COPY models/ models/

# FastAPI default port
EXPOSE 8000

# Healthcheck — Docker / ECS / compose can auto-restart unhealthy containers
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

# Run with 1 worker; adjust --workers for production if not behind Lambda
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
