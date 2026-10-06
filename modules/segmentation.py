"""
Module: Image Segmentation
Implements threshold-based segmentation, Otsu's bimodal thresholding,
and contour extraction with regional shape segmentation.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import cv2


def otsu_segmentation(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Performs Otsu's thresholding to find optimal threshold by minimizing intra-class variance.
    Returns segmented binary mask and calculated threshold value.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    # Apply Gaussian blur to reduce high frequency noise before Otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_val, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask, float(thresh_val)


def threshold_segmentation(img: np.ndarray, thresh_val: int = 128, invert: bool = False) -> np.ndarray:
    """
    Performs global threshold-based binary segmentation.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, thresh_val, 255, mode)
    return mask


def segment_and_extract_contours(img: np.ndarray, min_contour_area: int = 500) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Finds and draws contours on the input image.
    Extracts geometric properties of detected foreground regions.
    
    Returns:
        annotated_img: Image with colored bounding boxes and contour outlines.
        segmented_foreground: Masked foreground image.
        contour_info_list: List of dictionaries with contour properties.
    """
    img_rgb = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding for initial binary separation
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean mask using morphological closing & opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, hierarchy = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img_rgb.copy()
    contour_stats = []
    
    # Sort contours by area descending
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for idx, cnt in enumerate(sorted_contours):
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h) if (w * h) > 0 else 0
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Draw contour boundary (Cyan) and bounding box (Green)
        cv2.drawContours(annotated, [cnt], -1, (0, 255, 255), 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"R{idx+1}: {int(area)}px", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        contour_stats.append({
            "Region": f"Region {idx+1}",
            "Area (px)": int(area),
            "Perimeter (px)": round(perimeter, 2),
            "Bounding Box (x, y, w, h)": f"({x}, {y}, {w}, {h})",
            "Aspect Ratio": round(aspect_ratio, 3),
            "Extent": round(extent, 3),
            "Solidity": round(solidity, 3),
        })
        
    # Masked foreground extraction
    mask_3ch = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2RGB)
    foreground = cv2.bitwise_and(img_rgb, mask_3ch)
    
    return annotated, foreground, contour_stats
