"""
Module: Image Acquisition & Metadata Extraction
Handles loading, format validation, and image property extraction.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import io


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Loads an image from raw bytes into an RGB NumPy array.
    
    Args:
        image_bytes: Raw bytes of the uploaded image.
        
    Returns:
        RGB numpy array (H, W, C) or None if decoding fails.
    """
    try:
        # Decode using OpenCV from byte buffer
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback to PIL if OpenCV decoder fails
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return np.array(pil_img)
        # Convert BGR (OpenCV standard) to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"[Error] Failed to load image from bytes: {e}")
        return None


def load_image_from_path(file_path: str) -> Optional[np.ndarray]:
    """
    Loads an image from a local file path into an RGB NumPy array.
    
    Args:
        file_path: Path to the image file.
        
    Returns:
        RGB numpy array (H, W, C) or None if loading fails.
    """
    if not os.path.exists(file_path):
        return None
    try:
        img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil_img = Image.open(file_path).convert("RGB")
            return np.array(pil_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"[Error] Failed to load image from {file_path}: {e}")
        return None


def get_image_metadata(img: np.ndarray, file_size_bytes: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts essential academic IPCV image properties.
    
    Args:
        img: Input image as numpy array.
        file_size_bytes: Optional raw file size in bytes.
        
    Returns:
        Dictionary containing image properties and statistical properties.
    """
    if img is None:
        return {}

    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    total_pixels = height * width
    aspect_ratio = round(width / height, 3) if height > 0 else 0
    dtype_str = str(img.dtype)
    
    # Calculate channel statistics
    if channels == 3:
        mean_val = (round(float(np.mean(img[:, :, 0])), 2),
                    round(float(np.mean(img[:, :, 1])), 2),
                    round(float(np.mean(img[:, :, 2])), 2))
        std_val = (round(float(np.std(img[:, :, 0])), 2),
                   round(float(np.std(img[:, :, 1])), 2),
                   round(float(np.std(img[:, :, 2])), 2))
        color_model = "RGB (3-Channel Color)"
    else:
        mean_val = round(float(np.mean(img)), 2)
        std_val = round(float(np.std(img)), 2)
        color_model = "Grayscale (1-Channel)"

    min_val = int(np.min(img))
    max_val = int(np.max(img))
    
    # Estimate memory footprint
    memory_kb = round(img.nbytes / 1024, 2)
    file_size_kb = round(file_size_bytes / 1024, 2) if file_size_bytes else memory_kb

    return {
        "Width (Pixels)": width,
        "Height (Pixels)": height,
        "Channels": channels,
        "Color Representation": color_model,
        "Aspect Ratio (W:H)": aspect_ratio,
        "Total Pixels": total_pixels,
        "Data Type": dtype_str,
        "Pixel Intensity Range": f"[{min_val}, {max_val}]",
        "Mean Intensity (R, G, B)": mean_val,
        "Std Intensity (R, G, B)": std_val,
        "In-Memory Size (KB)": memory_kb,
        "File Size (KB)": file_size_kb,
    }
