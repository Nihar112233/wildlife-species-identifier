"""
Module: Morphological Operations
Implements mathematical morphology for binary and grayscale image processing.
"""

from typing import Tuple
import numpy as np
import cv2


def apply_morphological_operation(img: np.ndarray, operation: str, kernel_shape: str = "Rectangle",
                                  kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Applies mathematical morphology using configurable structuring elements.
    
    Operations:
    - Erosion: Shrinks foreground objects, eliminates small white noise.
    - Dilation: Expands foreground objects, joins broken parts.
    - Opening: Erosion followed by Dilation (removes noise while preserving shape).
    - Closing: Dilation followed by Erosion (closes small holes and bridges gaps).
    - Morphological Gradient: Dilation minus Erosion (extracts object boundaries).
    - Top Hat: Original minus Opening (highlights bright features smaller than kernel).
    - Black Hat: Closing minus Original (highlights dark features smaller than kernel).
    """
    # Map structuring element shapes
    shape_map = {
        "Rectangle": cv2.MORPH_RECT,
        "Ellipse": cv2.MORPH_ELLIPSE,
        "Cross": cv2.MORPH_CROSS
    }
    se_shape = shape_map.get(kernel_shape, cv2.MORPH_RECT)
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(se_shape, (k, k))
    
    # Process image (works on RGB or Grayscale)
    op_map = {
        "Erosion": cv2.MORPH_ERODE,
        "Dilation": cv2.MORPH_DILATE,
        "Opening": cv2.MORPH_OPEN,
        "Closing": cv2.MORPH_CLOSE,
        "Morphological Gradient": cv2.MORPH_GRADIENT,
        "Top Hat": cv2.MORPH_TOPHAT,
        "Black Hat": cv2.MORPH_BLACKHAT
    }
    
    if operation == "Erosion":
        return cv2.erode(img, kernel, iterations=iterations)
    elif operation == "Dilation":
        return cv2.dilate(img, kernel, iterations=iterations)
    elif operation in op_map:
        return cv2.morphologyEx(img, op_map[operation], kernel, iterations=iterations)
    else:
        return img
