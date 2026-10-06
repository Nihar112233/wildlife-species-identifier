"""
Module: Image Enhancement
Implements contrast enhancement, histogram equalization, CLAHE, and gamma correction.
"""

from typing import Tuple
import numpy as np
import cv2


def equalize_histogram(img: np.ndarray, method: str = "Standard Global", 
                       clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Performs histogram equalization to enhance global or local contrast.
    - Standard Global (Grayscale or YCrCb color space).
    - CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    if len(img.shape) == 2:
        # Grayscale
        if method == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            return clahe.apply(img)
        else:
            return cv2.equalizeHist(img)
            
    # Color image: Transform to YCrCb space, equalize Y (luminance) channel, convert back to RGB
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    if method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        y_eq = clahe.apply(y)
    else:
        y_eq = cv2.equalizeHist(y)
        
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def apply_gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applies power-law (Gamma) transformation:
    V_out = 255 * (V_in / 255) ** gamma
    Gamma < 1.0 brightens shadows / dark regions.
    Gamma > 1.0 darkens highlights.
    """
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjusts contrast (alpha) and brightness (beta):
    g(x,y) = alpha * f(x,y) + beta
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
