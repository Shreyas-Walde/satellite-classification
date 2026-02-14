"""
Satellite Image Classifier API
FastAPI app with AWS Lambda compatibility via Mangum.
"""
import io
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms


# ============================================
# MODEL ARCHITECTURE (from training notebook)
# ============================================

def conv_block(in_channels, out_channels, pool=False):
    """Convolutional block with optional pooling"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ImageClassificationBase(nn.Module):
    """Base class for image classification"""
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}


class ResNet9(ImageClassificationBase):
    """ResNet9 - EXACT architecture from training"""
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# ============================================
# CONFIGURATION
# ============================================

CLASS_NAMES = ['cloudy', 'desert', 'green_area', 'water']
MODEL_PATH = 'models/resnet50_satellite_model.pth'
IMAGE_SIZE = 64  # Must match training: transforms.Resize(64)
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Transform — MUST match training pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


# ============================================
# LIFESPAN (modern replacement for on_event)
# ============================================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model

    print("=" * 60)
    print("🚀 Loading Satellite Classifier Model...")
    print("=" * 60)

    try:
        model = ResNet9(in_channels=3, num_classes=len(CLASS_NAMES))
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {CLASS_NAMES}")
        print(f"🖼️  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print("=" * 60)
        print("✅ Server ready! Open http://127.0.0.1:8000/docs")
        print("=" * 60)

    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print("💡 Make sure your .pth file is in the models/ folder")
        raise
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        raise

    yield  # App runs here

    # Cleanup on shutdown
    model = None
    print("🛑 Server shutting down, model unloaded.")


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="🛰️ Satellite Image Classifier",
    description=(
        "Classifies satellite images into 4 categories: "
        "cloudy, desert, green_area, water. "
        "Upload an image to /predict to get results."
    ),
    version="1.0.0",
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
    """Root endpoint — API info"""
    return {
        "message": "🛰️ Satellite Image Classifier API",
        "status": "running",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a satellite image.

    Upload a JPG/PNG satellite image and receive:
    - predicted class (cloudy / desert / green_area / water)
    - confidence score (0-1)
    - probabilities for all classes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate file extension
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    try:
        # 1. Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 2. Preprocess (must match training!)
        image_tensor = transform(image).unsqueeze(0)

        # 3. Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 4. Prepare results
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = float(confidence.item())

        all_probabilities = {
            CLASS_NAMES[i]: round(float(probabilities[0][i].item()), 4)
            for i in range(len(CLASS_NAMES))
        }

        print(f"✅ Prediction: {predicted_class} ({confidence_score:.2%})")

        return {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence_score, 4),
                "confidence_percent": f"{confidence_score:.1%}",
                "all_probabilities": all_probabilities,
            },
            "filename": file.filename,
        }

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================
# AWS LAMBDA HANDLER (via Mangum)
# ============================================

try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    # Mangum not installed — running locally, that's fine
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
