"""
Module: Image Preprocessing & Filtering
Implements classical IPCV algorithms for resizing, color conversion, normalization,
noise reduction, sharpening, thresholding, and edge detection.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def resize_image(img: np.ndarray, target_width: int, target_height: int, interpolation_type: str = "Linear") -> np.ndarray:
    """
    Resizes an image using specified OpenCV interpolation methods.
    """
    interp_map = {
        "Nearest Neighbor": cv2.INTER_NEAREST,
        "Linear": cv2.INTER_LINEAR,
        "Cubic": cv2.INTER_CUBIC,
        "Area": cv2.INTER_AREA,
        "Lanczos4": cv2.INTER_LANCZOS4
    }
    flag = interp_map.get(interpolation_type, cv2.INTER_LINEAR)
    return cv2.resize(img, (target_width, target_height), interpolation=flag)


def convert_color_space(img: np.ndarray, target_space: str) -> np.ndarray:
    """
    Converts RGB image to Grayscale, HSV, LAB, or YCrCb color space.
    """
    if len(img.shape) == 2:
        # Grayscale input
        if target_space == "RGB":
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    
    if target_space == "Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif target_space == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif target_space == "LAB":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif target_space == "YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif target_space == "RGB":
        return img.copy()
    else:
        raise ValueError(f"Unsupported color space: {target_space}")


def normalize_image(img: np.ndarray, method: str = "Min-Max") -> np.ndarray:
    """
    Normalizes image pixel intensities for mathematical modeling.
    - Min-Max: Scales intensities to [0, 1] range (returned as float32, or scaled back to [0, 255] uint8 for display).
    - Z-Score: Standardizes intensities (zero mean, unit variance).
    """
    img_float = img.astype(np.float32)
    
    if method == "Min-Max [0, 1]":
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        if max_val > min_val:
            normalized = (img_float - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(img_float)
        return (normalized * 255).astype(np.uint8)
        
    elif method == "Z-Score Standardization":
        mean = np.mean(img_float)
        std = np.std(img_float)
        if std > 0:
            standardized = (img_float - mean) / std
        else:
            standardized = np.zeros_like(img_float)
        # Shift and scale to displayable uint8 [0, 255]
        clipped = np.clip((standardized * 32) + 128, 0, 255)
        return clipped.astype(np.uint8)
        
    return img


def apply_noise_reduction(img: np.ndarray, filter_type: str, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies spatial filtering for noise reduction:
    - Gaussian Filter: Weighted averaging based on 2D Gaussian distribution.
    - Median Filter: Non-linear filter effective for salt-and-pepper noise.
    - Box (Average) Filter: Uniform averaging kernel.
    - Bilateral Filter: Edge-preserving smoothing filter.
    """
    # Ensure kernel size is odd
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    if filter_type == "Gaussian Filter":
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    elif filter_type == "Median Filter":
        return cv2.medianBlur(img, k)
    elif filter_type == "Averaging (Box) Filter":
        return cv2.blur(img, (k, k))
    elif filter_type == "Bilateral Filter (Edge-Preserving)":
        d = k
        sigma_color = sigma * 25
        sigma_space = sigma * 25
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    else:
        return img


def apply_sharpening(img: np.ndarray, method: str = "Unsharp Masking", strength: float = 1.5) -> np.ndarray:
    """
    Enhances high-frequency spatial components (edges/details).
    - 3x3 Kernel Sharpening: Convolution with classical high-pass sharpening kernel.
    - Laplacian Sharpening: Enhances second derivative rapid intensity changes.
    - Unsharp Masking: I_sharp = I + alpha * (I - I_blurred).
    """
    if method == "3x3 Sharpening Kernel":
        # Classical 3x3 kernel: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        kernel = np.array([[0, -1, 0],
                           [-1, 4 + strength, -1],
                           [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    elif method == "Laplacian Sharpening":
        # Compute Laplacian and add to original
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpened = img.astype(np.float64) - (strength * 0.5) * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    elif method == "Unsharp Masking":
        # Unsharp masking algorithm
        gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        unsharp_mask = cv2.addWeighted(img, 1.0 + strength, gaussian, -strength, 0)
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        
    return img


def apply_thresholding(img: np.ndarray, threshold_type: str, thresh_val: int = 127, max_val: int = 255,
                       block_size: int = 11, c_constant: int = 2) -> np.ndarray:
    """
    Applies global or adaptive intensity thresholding.
    Converts RGB to grayscale automatically if needed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # Ensure block size is odd and > 1 for adaptive
    b_size = block_size if block_size % 2 == 1 else block_size + 1
    b_size = max(3, b_size)

    if threshold_type == "Binary Threshold":
        _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
    elif threshold_type == "Binary Inverted":
        _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY_INV)
    elif threshold_type == "Truncate":
        _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_TRUNC)
    elif threshold_type == "To Zero":
        _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_TOZERO)
    elif threshold_type == "To Zero Inverted":
        _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_TOZERO_INV)
    elif threshold_type == "Adaptive Mean":
        result = cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b_size, c_constant)
    elif threshold_type == "Adaptive Gaussian":
        result = cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, c_constant)
    else:
        result = gray
        
    return result


def apply_edge_detection(img: np.ndarray, method: str, low_thresh: int = 50, high_thresh: int = 150, 
                         sobel_ksize: int = 3) -> np.ndarray:
    """
    Applies classical edge detection operators:
    - Canny Edge Detector: Gradient computation + Non-maximum suppression + Hysteresis thresholding.
    - Sobel Operator: Discrete differentiation in X, Y, or Combined magnitude.
    - Laplacian: Second-order isotropic derivative.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    k = sobel_ksize if sobel_ksize % 2 == 1 else sobel_ksize + 1
    
    if method == "Canny Edge Detection":
        return cv2.Canny(gray, low_thresh, high_thresh)
        
    elif method == "Sobel X (Vertical Edges)":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        abs_sobelx = np.absolute(sobelx)
        return np.uint8(np.clip(abs_sobelx, 0, 255))
        
    elif method == "Sobel Y (Horizontal Edges)":
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        abs_sobely = np.absolute(sobely)
        return np.uint8(np.clip(abs_sobely, 0, 255))
        
    elif method == "Sobel Combined Magnitude":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(mag, 0, 255))
        
    elif method == "Laplacian Operator":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
        abs_lap = np.absolute(lap)
        return np.uint8(np.clip(abs_lap, 0, 255))
        
    return gray
