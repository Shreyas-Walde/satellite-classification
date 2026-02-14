"""
Configuration settings for the satellite classifier API
"""
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "resnet50_satellite_model.pth"

# Model configuration
NUM_CLASSES = 4
INPUT_CHANNELS = 3
IMAGE_SIZE = 64  # CRITICAL: Must match training! (From notebook: Resize(64))

# Class names - MUST match training order
CLASS_NAMES = [
    "cloudy",
    "desert", 
    "green_area",
    "water"
]

# API configuration
API_TITLE = "Satellite Image Classifier"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
🛰️ Satellite Image Classification API

Classifies satellite images into 4 categories:
- **Cloudy**: Cloud-covered areas
- **Desert**: Desert/arid regions  
- **Green Area**: Vegetation/forests
- **Water**: Water bodies

Built with ResNet9 architecture and PyTorch.
"""

# Inference settings
DEVICE = "cpu"  # Change to "cuda" if GPU available
MAX_IMAGE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Logging
LOG_LEVEL = "INFO"