# 🛰️ Satellite Image Classifier

**An end-to-end deep learning project** — from raw satellite imagery to a production-ready API deployed on AWS Lambda.

Classifies satellite images into **4 terrain categories**: `cloudy`, `desert`, `green_area`, `water`

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.17-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/AWS_Lambda-Deployed-FF9900?logo=awslambda&logoColor=white)

---

## ✨ Highlights

- **Custom ResNet9** model trained on the EuroSAT dataset (~5,600 images)
- **Dual inference engine** — switch between PyTorch and ONNX Runtime on the fly
- **OpenCV + NumPy preprocessing** — no torchvision dependency at inference time
- **Multi-stage Docker build** for a slim production image
- **AWS Lambda-ready** with a dedicated ONNX-only Dockerfile (~416 MB)
- **FastAPI** with Swagger UI, health checks, and CORS middleware

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BUILD PIPELINE                               │
│                                                                     │
│  EuroSAT Dataset ──► Training (ResNet9) ──► model.pth ──► model.onnx│
│  (5,631 images)      (Notebook)             (26 MB)      (25 MB)   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVING PIPELINE                              │
│                                                                     │
│  Client ──► API Gateway ──► AWS Lambda ──► FastAPI ──► ONNX Runtime │
│         (HTTP POST)       (Docker image)  (Mangum)   (model.onnx)  │
│                                                                     │
│  OR locally:                                                        │
│  Client ──► Uvicorn ──► FastAPI ──► PyTorch / ONNX (selectable)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
satelite/
├── main.py                    # FastAPI application (dual-engine inference + Lambda handler)
├── model.py                   # ResNet9 architecture definition (must match training)
├── preprocessing.py           # Image preprocessing with OpenCV + NumPy
├── satellite_classifier.py    # Initial training prototype (ResNet18, quick experiments)
│
├── models/
│   ├── resnet50_satellite_model.pth   # Trained PyTorch weights (~26 MB)
│   └── model.onnx                     # Exported ONNX model (~25 MB)
│
├── notebooks/
│   ├── 01_quick_test.ipynb    # Full training pipeline (data → train → evaluate)
│   └── test.ipynb             # Data exploration and visualization
│
├── scripts/
│   ├── export_onnx.py         # PyTorch → ONNX export with validation
│   └── benchmark.py           # PyTorch vs ONNX Runtime performance comparison
│
├── data/                      # EuroSAT satellite images (gitignored)
│   ├── cloudy/
│   ├── desert/
│   ├── green_area/
│   └── water/
│
├── Dockerfile                 # Multi-stage production build (PyTorch + ONNX)
├── Dockerfile.lambda          # AWS Lambda optimized build (ONNX-only, ~416 MB)
├── requirements.txt           # Production dependencies
├── requirements-lambda.txt    # Lambda-specific dependencies (no torch/uvicorn)
├── pyproject.toml             # Project metadata and dependency management
├── .dockerignore              # Files excluded from Docker builds
└── .gitignore                 # Files excluded from version control
```

---

## 🚀 The Build Journey

This section walks through every phase of the project, from raw data to a deployed cloud API.

### Phase 1 — Data Exploration & Preparation

**Dataset**: [EuroSAT](https://github.com/phelber/eurosat) — satellite images from the Sentinel-2 satellite

| Detail | Value |
|--------|-------|
| Total images | 5,631 |
| Classes | 4 (`cloudy`, `desert`, `green_area`, `water`) |
| Image size | Resized to 64×64 px |
| Train/Val split | 80% / 20% (4,504 / 1,127) |

**What I did:**
1. Downloaded the EuroSAT dataset and organized it into `data/` with subdirectories per class
2. Used `torchvision.datasets.ImageFolder` to automatically map folder names → class labels
3. Applied transforms: `Resize(64)` + `ToTensor()` (normalizes pixel values to [0, 1])
4. Split the data 80/20 using `torch.utils.data.random_split`
5. Created `DataLoader`s with batch size 32 and parallel data loading

**Files:** `notebooks/01_quick_test.ipynb`, `notebooks/test.ipynb`

---

### Phase 2 — Model Architecture: Custom ResNet9

Instead of using a massive pretrained ResNet50, I built a **lightweight custom ResNet9** — fast to train, small to deploy, and perfectly suited for 64×64 satellite images.

```
Input (3×64×64)
    │
    ├── Conv Block 1 ──► 64 channels
    ├── Conv Block 2 ──► 128 channels + MaxPool ─┐
    │   └── Residual Block (128→128→128)  ◄──────┘  ← skip connection
    ├── Conv Block 3 ──► 256 channels + MaxPool
    ├── Conv Block 4 ──► 512 channels + MaxPool ─┐
    │   └── Residual Block (512→512→512)  ◄──────┘  ← skip connection
    │
    ├── AdaptiveMaxPool → Flatten → Dropout(0.2)
    └── Linear(512 → 4 classes)

Output: [cloudy, desert, green_area, water]
```

| Metric | Value |
|--------|-------|
| Parameters | ~6.5M |
| Input shape | (batch, 3, 64, 64) |
| Output | 4 class logits |

Each **conv block** = `Conv2d(3×3)` → `BatchNorm2d` → `ReLU` (± `MaxPool2d`)

The **residual (skip) connections** allow gradients to flow cleanly through the network, preventing vanishing gradients even in deeper architectures.

**File:** `model.py`

> **Note:** The initial prototype in `satellite_classifier.py` used a pretrained ResNet18 for quick experiments. The final model is a custom ResNet9 for better control and smaller footprint.

---

### Phase 3 — Training

Training was done interactively in the Jupyter notebook with a structured training loop:

```python
# Training configuration
optimizer = Adam
scheduler = OneCycleLR
loss_fn   = CrossEntropyLoss
epochs    = multiple passes until convergence
batch     = 32
```

**Training loop flow:**
1. Forward pass → compute cross-entropy loss
2. Backward pass → compute gradients
3. Optimizer step → update weights
4. Track training loss + validation accuracy per epoch
5. Learning rate scheduling with OneCycleLR (ramps up, then decays — better convergence)

The trained weights are saved as `models/resnet50_satellite_model.pth`.

**File:** `notebooks/01_quick_test.ipynb`

---

### Phase 4 — ONNX Export & Optimization

**Why ONNX?**
| | PyTorch | ONNX Runtime |
|---|---|---|
| Purpose | Training + inference | Inference only |
| Install size | ~900 MB | ~30 MB |
| Speed | Baseline | Up to 2× faster |
| Dependency | Full framework | Lightweight runtime |

ONNX (Open Neural Network Exchange) is a standard format that lets you run models without the full PyTorch framework — critical for keeping Lambda images small.

**Export process** (`scripts/export_onnx.py`):
1. Load the ResNet9 architecture + trained weights
2. Create a dummy input tensor `(1, 3, 64, 64)`
3. Trace the model using `torch.onnx.export()` with opset 13
4. Enable `dynamic_axes` for variable batch sizes
5. Validate with `onnx.checker.check_model()`
6. Cross-verify: run the same input through both engines and confirm outputs match (< 1e-5 difference)

```bash
python scripts/export_onnx.py
# Output: models/model.onnx (25 MB)
```

---

### Phase 5 — Preprocessing with OpenCV + NumPy

Replaced PIL/torchvision with **OpenCV + NumPy** for inference preprocessing. This eliminates the torchvision dependency and is the industry standard for computer vision pipelines.

**Pipeline** (`preprocessing.py`):

```
Raw bytes ──► cv2.imdecode() ──► BGR array
                                    │
                             cv2.cvtColor() ──► RGB
                                    │
                             cv2.resize() ──► 64×64
                                    │
                             / 255.0 ──► [0, 1] float32
                                    │
                             np.transpose() ──► CHW format
                                    │
                             np.expand_dims() ──► (1, 3, 64, 64)
```

Also implemented a **pure NumPy softmax** to convert raw model logits into probability distributions — no PyTorch needed at inference time.

---

### Phase 6 — FastAPI with Dual Inference Engine

The API (`main.py`) supports **both** PyTorch and ONNX Runtime, selectable per request:

```
POST /predict?engine=onnx      ← fast, lightweight (default)
POST /predict?engine=pytorch    ← original, full framework
```

**Key features:**
- **Lifespan** model loading — both engines loaded at startup, cleaned up on shutdown
- **Automatic fallback** — if ONNX isn't available, falls back to PyTorch
- **File validation** — checks extensions (`.jpg`, `.png`, `.bmp`, `.tiff`, `.webp`)
- **CORS middleware** — allows cross-origin requests from any frontend
- **Mangum handler** — makes the same FastAPI app run on AWS Lambda with zero code changes

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info and engine status |
| `GET` | `/health` | Health check (which engines are loaded) |
| `POST` | `/predict` | Classify a satellite image |
| `GET` | `/docs` | Swagger UI (auto-generated) |

**Example response** from `POST /predict`:
```json
{
  "success": true,
  "prediction": {
    "class": "green_area",
    "confidence": 0.9847,
    "confidence_percent": "98.5%",
    "all_probabilities": {
      "cloudy": 0.0021,
      "desert": 0.0089,
      "green_area": 0.9847,
      "water": 0.0043
    }
  },
  "engine": "onnx",
  "inference_time_ms": 12.34,
  "filename": "satellite_image.jpg"
}
```

---

### Phase 7 — Benchmarking PyTorch vs ONNX

The benchmark script (`scripts/benchmark.py`) runs a quantitative comparison:

```bash
python scripts/benchmark.py
```

It performs:
- 5 warmup iterations (exclude cold-start from measurements)
- 50 timed inference runs per engine
- Reports: average, median, min, max, std deviation latency
- Computes speedup factor
- Verifies outputs match between engines

```
═════════════════════════════════════════════════════════════════
                           RESULTS
═════════════════════════════════════════════════════════════════

Metric                     PyTorch     ONNX Runtime
---------------------------------------------------------
Predicted class             desert           desert
Avg latency (ms)           15.23ms          7.41ms
Median latency (ms)        14.89ms          6.92ms

⚡ ONNX is ~2x faster than PyTorch
```

---

### Phase 8 — Docker Containerization

#### Production Dockerfile (Multi-stage)

`Dockerfile` — Full build with both engines:

```
Stage 1 (builder)          Stage 2 (runtime)
┌──────────────────┐      ┌──────────────────────┐
│ python:3.12-slim │      │ python:3.12-slim     │
│ + gcc            │      │ + libgl1, libglib2.0 │
│ + pip install    │─────►│ + pre-built packages │
│   (torch CPU,    │ COPY │ + app code           │
│    requirements) │      │ + model weights      │
└──────────────────┘      │ + healthcheck        │
                          └──────────────────────┘
```

**Why multi-stage?**
- Builder stage has gcc and pip cache — not needed at runtime
- Final image is clean and small
- CPU-only PyTorch avoids pulling ~2 GB of CUDA libraries

```bash
# Build
docker build -t satellite-classifier .

# Run
docker run -p 8000:8000 satellite-classifier

# Test
curl http://localhost:8000/health
```

#### Lambda Dockerfile (ONNX-only)

`Dockerfile.lambda` — Optimized for AWS Lambda (~416 MB total):

```bash
# Build
docker build -f Dockerfile.lambda -t satellite-classifier:lambda .

# Test locally (Lambda emulator)
docker run -p 9000:8080 satellite-classifier:lambda
```

Key differences from the production Dockerfile:
- Uses `public.ecr.aws/lambda/python:3.12` base image
- **No PyTorch** — ONNX Runtime only (saves ~900 MB)
- **No uvicorn** — Lambda handles HTTP
- Ships only `model.onnx` (skips the `.pth` file)

---

### Phase 9 — AWS Lambda Deployment

The final deployment uses **ECR + Lambda + API Gateway**:

```
┌──────────┐     ┌─────────────┐     ┌────────────┐     ┌──────────┐
│  Client  │────►│ API Gateway │────►│ AWS Lambda │────►│ FastAPI  │
│          │◄────│ (HTTP API)  │◄────│ (Docker)   │◄────│ + ONNX   │
└──────────┘     └─────────────┘     └────────────┘     └──────────┘
```

**Deployment steps (AWS Console):**

1. **ECR** — Create a private repository `satellite-classifier`, push the Lambda Docker image
2. **IAM** — Create a role with `AWSLambdaBasicExecutionRole` policy
3. **Lambda** — Create function from container image (1024 MB RAM, 60s timeout)
4. **API Gateway** — Create HTTP API with `ANY /{proxy+}` route → Lambda integration
5. **Test** — Hit the invoke URL `/health`, `/docs`, and `/predict`

> ⚠️ **Cold starts**: First invocation after idle takes 10-20 seconds (container spin-up + ONNX model loading). Subsequent calls are fast (~100-200ms). Consider Provisioned Concurrency or a scheduled ping to keep the function warm.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker (for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Shreyas-Walde/satellite-classification.git
cd satellite-classification

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run the API

```bash
# Start the development server
python main.py

# Server starts at http://127.0.0.1:8000
# Swagger UI at http://127.0.0.1:8000/docs
```

### Test it

```bash
# Health check
curl http://localhost:8000/health

# Classify an image (using ONNX — default)
curl -X POST http://localhost:8000/predict \
  -F "file=@data/cloudy/cloudy_001.jpg"

# Classify with PyTorch engine
curl -X POST "http://localhost:8000/predict?engine=pytorch" \
  -F "file=@data/desert/desert_001.jpg"
```

---

## 🐳 Docker

```bash
# --- Production (both engines) ---
docker build -t satellite-classifier .
docker run -p 8000:8000 satellite-classifier

# --- Lambda (ONNX-only) ---
docker build -f Dockerfile.lambda -t satellite-classifier:lambda .
docker run -p 9000:8080 satellite-classifier:lambda
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.12 | Core language |
| **Deep Learning** | PyTorch | Model training + inference engine |
| **Model Format** | ONNX | Lightweight inference (no PyTorch needed) |
| **Runtime** | ONNX Runtime | Fast CPU inference |
| **Computer Vision** | OpenCV | Image preprocessing |
| **Numerical** | NumPy | Array operations, softmax |
| **API Framework** | FastAPI | REST API with auto-docs |
| **ASGI Server** | Uvicorn | Local development server |
| **Lambda Adapter** | Mangum | FastAPI → AWS Lambda bridge |
| **Containerization** | Docker | Reproducible builds |
| **Cloud** | AWS Lambda + ECR + API Gateway | Serverless deployment |
| **Package Manager** | uv | Fast Python dependency management |

---

## 📜 License

This project is for educational and portfolio purposes.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Shreyas-Walde">Shreyas Walde</a>
</p>
