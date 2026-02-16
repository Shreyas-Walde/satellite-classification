"""
Satellite Image Classifier API

Dual inference engine:
  - PyTorch (original, full framework)
  - ONNX Runtime (optimized, lightweight)

Uses OpenCV + NumPy for preprocessing (no torchvision dependency for inference).
Compatible with AWS Lambda via Mangum.
"""
import io
import os
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import preprocess_image, softmax


# ============================================
# CONFIGURATION
# ============================================

CLASS_NAMES = ["cloudy", "desert", "green_area", "water"]
MODEL_DIR = Path("models")
PTH_PATH = MODEL_DIR / "resnet50_satellite_model.pth"
ONNX_PATH = MODEL_DIR / "model.onnx"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class InferenceEngine(str, Enum):
    pytorch = "pytorch"
    onnx = "onnx"


# ============================================
# GLOBAL MODEL STATE
# ============================================

pytorch_model = None
onnx_session = None


# ============================================
# LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global pytorch_model, onnx_session

    print("=" * 60)
    print("🚀 Loading Satellite Classifier Models...")
    print("=" * 60)

    # --- Load PyTorch model ---
    try:
        import torch
        from model import ResNet9

        pytorch_model = ResNet9(in_channels=3, num_classes=len(CLASS_NAMES))
        checkpoint = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            pytorch_model.load_state_dict(checkpoint)

        pytorch_model.eval()
        print("✅ PyTorch model loaded")
    except Exception as e:
        print(f"⚠️  PyTorch model failed to load: {e}")
        pytorch_model = None

    # --- Load ONNX model ---
    try:
        import onnxruntime as ort

        if ONNX_PATH.exists():
            onnx_session = ort.InferenceSession(
                str(ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
            print("✅ ONNX model loaded")
        else:
            print(f"⚠️  ONNX model not found at {ONNX_PATH}")
            print("   Run: python scripts/export_onnx.py")
    except Exception as e:
        print(f"⚠️  ONNX model failed to load: {e}")
        onnx_session = None

    # --- Status ---
    default = "onnx" if onnx_session else ("pytorch" if pytorch_model else "none")
    print(f"\n📊 Classes: {CLASS_NAMES}")
    print(f"🔧 Default engine: {default}")
    print("=" * 60)
    print("✅ Server ready! Open http://127.0.0.1:8000/docs")
    print("=" * 60)

    yield

    pytorch_model = None
    onnx_session = None
    print("🛑 Server shut down.")


# ============================================
# INFERENCE FUNCTIONS
# ============================================

def predict_pytorch(image_array: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Run inference with PyTorch.
    
    Args:
        image_array: Preprocessed image (1, 3, 64, 64) as numpy
    
    Returns:
        (probabilities, inference_time_ms)
    """
    import torch

    tensor = torch.from_numpy(image_array)

    start = time.perf_counter()
    with torch.no_grad():
        output = pytorch_model(tensor).numpy()
    elapsed = (time.perf_counter() - start) * 1000

    probs = softmax(output)
    return probs, elapsed


def predict_onnx(image_array: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Run inference with ONNX Runtime.
    
    Args:
        image_array: Preprocessed image (1, 3, 64, 64) as numpy
    
    Returns:
        (probabilities, inference_time_ms)
    """
    start = time.perf_counter()
    output = onnx_session.run(None, {"input": image_array})[0]
    elapsed = (time.perf_counter() - start) * 1000

    probs = softmax(output)
    return probs, elapsed


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="🛰️ Satellite Image Classifier",
    description=(
        "Classifies satellite images into 4 categories: "
        "cloudy, desert, green_area, water.\n\n"
        "Supports **dual inference engines**: PyTorch (original) and ONNX Runtime (optimized).\n\n"
        "Use `?engine=onnx` or `?engine=pytorch` to select."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API info and engine status."""
    return {
        "message": "🛰️ Satellite Image Classifier API",
        "version": "2.0.0",
        "engines": {
            "pytorch": "loaded" if pytorch_model is not None else "not loaded",
            "onnx": "loaded" if onnx_session is not None else "not loaded",
        },
        "classes": CLASS_NAMES,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check with engine status."""
    return {
        "status": "healthy",
        "pytorch_loaded": pytorch_model is not None,
        "onnx_loaded": onnx_session is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    engine: InferenceEngine = Query(
        default=InferenceEngine.onnx,
        description="Inference engine: 'onnx' (fast, lightweight) or 'pytorch' (original)",
    ),
):
    """
    Classify a satellite image.

    Upload a JPG/PNG satellite image and receive:
    - Predicted class (cloudy / desert / green_area / water)
    - Confidence score (0-1)
    - Probabilities for all classes
    - Inference time in milliseconds
    - Which engine processed the request
    """
    # Validate engine availability
    if engine == InferenceEngine.pytorch and pytorch_model is None:
        raise HTTPException(503, "PyTorch engine not loaded")
    if engine == InferenceEngine.onnx and onnx_session is None:
        # Fallback to PyTorch if ONNX not available
        if pytorch_model is not None:
            engine = InferenceEngine.pytorch
        else:
            raise HTTPException(503, "No inference engine available")

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    try:
        # 1. Read image bytes
        image_bytes = await file.read()

        # 2. Preprocess with OpenCV + NumPy
        image_array = preprocess_image(image_bytes)

        # 3. Run inference with selected engine
        if engine == InferenceEngine.pytorch:
            probs, inference_ms = predict_pytorch(image_array)
        else:
            probs, inference_ms = predict_onnx(image_array)

        # 4. Extract results
        predicted_idx = int(np.argmax(probs[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[0][predicted_idx])

        all_probabilities = {
            CLASS_NAMES[i]: round(float(probs[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        print(f"✅ [{engine.value}] {predicted_class} ({confidence:.1%}) in {inference_ms:.1f}ms")

        return {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "confidence_percent": f"{confidence:.1%}",
                "all_probabilities": all_probabilities,
            },
            "engine": engine.value,
            "inference_time_ms": round(inference_ms, 2),
            "filename": file.filename,
        }

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ============================================
# AWS LAMBDA HANDLER
# ============================================

try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = None


# ============================================
# LOCAL DEV SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n🚀 Starting server...")
    print("📖 Docs: http://127.0.0.1:8000/docs\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
