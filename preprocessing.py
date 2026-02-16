"""
Image Preprocessing with OpenCV + NumPy

Why OpenCV instead of PIL/torchvision?
    - OpenCV is the industry standard for computer vision
    - NumPy arrays work directly with ONNX Runtime (no torch dependency)
    - Faster than PIL for image operations
    - Shows OpenCV + NumPy proficiency (JD requirement)

Pipeline:
    1. Read image bytes → OpenCV BGR array
    2. Convert BGR → RGB
    3. Resize to 64x64 (must match training)
    4. Normalize to [0, 1] float32
    5. Reshape to (1, 3, H, W) — batch format
"""
import numpy as np
import cv2


# Must match training configuration
IMAGE_SIZE = 64


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a model-ready numpy array.
    
    Args:
        image_bytes: Raw image file bytes (from upload or file read)
    
    Returns:
        np.ndarray of shape (1, 3, 64, 64), dtype float32, range [0, 1]
    
    Pipeline matches training transforms:
        transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),  # Converts to [0,1] and CHW format
        ])
    """
    # Step 1: Decode image bytes → OpenCV array (BGR, uint8)
    # np.frombuffer creates a 1D array of bytes
    # cv2.imdecode converts it to a 3D image array (H, W, C)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image. File may be corrupted or not a valid image.")
    
    # Step 2: BGR → RGB
    # OpenCV loads images in BGR order, but our model was trained on RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 3: Resize to IMAGE_SIZE x IMAGE_SIZE
    # cv2.INTER_LINEAR = bilinear interpolation (same as PIL's default)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Step 4: Normalize to [0, 1] float32
    # torchvision.transforms.ToTensor() divides by 255
    img = img.astype(np.float32) / 255.0
    
    # Step 5: HWC → CHW (Height, Width, Channels → Channels, Height, Width)
    # PyTorch/ONNX expect channel-first format: (C, H, W)
    # NumPy/OpenCV use channel-last: (H, W, C)
    img = np.transpose(img, (2, 0, 1))  # (64, 64, 3) → (3, 64, 64)
    
    # Step 6: Add batch dimension
    # Model expects (batch_size, 3, 64, 64)
    img = np.expand_dims(img, axis=0)  # (3, 64, 64) → (1, 3, 64, 64)
    
    return img


def preprocess_image_from_path(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image from a file path.
    Useful for testing and benchmarking.
    """
    with open(image_path, "rb") as f:
        return preprocess_image(f.read())


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities from raw model logits.
    
    In PyTorch: torch.nn.functional.softmax(output, dim=1)
    In NumPy: manual implementation
    
    Args:
        x: Raw model output of shape (1, num_classes)
    
    Returns:
        Probability distribution (sums to 1.0)
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
