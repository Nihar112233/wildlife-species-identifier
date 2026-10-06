"""
Module: Feature Extraction
Extracts color histograms, statistical moments, GLCM texture descriptors,
geometric shape features, and edge-based descriptors.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew


def extract_color_features(img: np.ndarray) -> Dict[str, Any]:
    """
    Computes statistical color moments and channel histograms for RGB and HSV.
    """
    features = {}
    
    # 1. RGB Statistical Moments
    r_chan, g_chan, b_chan = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    features["Mean_R"] = round(float(np.mean(r_chan)), 2)
    features["Mean_G"] = round(float(np.mean(g_chan)), 2)
    features["Mean_B"] = round(float(np.mean(b_chan)), 2)
    
    features["Std_R"] = round(float(np.std(r_chan)), 2)
    features["Std_G"] = round(float(np.std(g_chan)), 2)
    features["Std_B"] = round(float(np.std(b_chan)), 2)
    
    features["Skew_R"] = round(float(skew(r_chan.ravel())), 3)
    features["Skew_G"] = round(float(skew(g_chan.ravel())), 3)
    features["Skew_B"] = round(float(skew(b_chan.ravel())), 3)
    
    # 2. HSV Statistical Moments
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    features["Mean_Hue"] = round(float(np.mean(h_chan)), 2)
    features["Mean_Sat"] = round(float(np.mean(s_chan)), 2)
    features["Mean_Val"] = round(float(np.mean(v_chan)), 2)
    
    return features


def compute_color_histograms(img: np.ndarray, bins: int = 32) -> Dict[str, np.ndarray]:
    """
    Computes 1D normalized color histograms for R, G, B channels and H, S, V channels.
    """
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256]).ravel()
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256]).ravel()
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256]).ravel()
    
    # Normalize histograms
    hist_r /= np.sum(hist_r) + 1e-7
    hist_g /= np.sum(hist_g) + 1e-7
    hist_b /= np.sum(hist_b) + 1e-7
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).ravel()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).ravel()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).ravel()
    
    hist_h /= np.sum(hist_h) + 1e-7
    hist_s /= np.sum(hist_s) + 1e-7
    hist_v /= np.sum(hist_v) + 1e-7
    
    return {
        "Red": hist_r,
        "Green": hist_g,
        "Blue": hist_b,
        "Hue": hist_h,
        "Saturation": hist_s,
        "Value": hist_v
    }


def extract_glcm_texture_features(img: np.ndarray, distances: List[int] = [1], 
                                  angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, float]:
    """
    Computes Haralick Texture features from Gray-Level Co-occurrence Matrix (GLCM).
    - Contrast: Local intensity variations.
    - Dissimilarity: Linear difference in gray levels.
    - Homogeneity: Closeness of element distribution to diagonal.
    - Energy: Uniformity / Angular Second Moment.
    - Correlation: Linear dependency of gray levels.
    - ASM: Sum of squared elements in GLCM.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # Compute GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    asm = float(np.mean(graycoprops(glcm, 'ASM')))
    
    return {
        "GLCM_Contrast": round(contrast, 4),
        "GLCM_Dissimilarity": round(dissimilarity, 4),
        "GLCM_Homogeneity": round(homogeneity, 4),
        "GLCM_Energy": round(energy, 4),
        "GLCM_Correlation": round(correlation, 4),
        "GLCM_ASM": round(asm, 6),
    }


def extract_shape_features(img: np.ndarray) -> Dict[str, Any]:
    """
    Extracts geometric and invariant shape descriptors from the primary foreground contour.
    - Area, Perimeter, Circularity (Compactness), Aspect Ratio, Extent, Solidity, Hu Moments.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            "Contour_Area": 0, "Perimeter": 0, "Circularity": 0, "Aspect_Ratio": 0,
            "Extent": 0, "Solidity": 0, "Hu_1": 0, "Hu_2": 0, "Hu_3": 0
        }
        
    # Pick largest contour
    largest_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_cnt)
    perimeter = cv2.arcLength(largest_cnt, True)
    
    # Circularity: 4 * pi * Area / Perimeter^2
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    x, y, w, h = cv2.boundingRect(largest_cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = float(area) / (w * h) if (w * h) > 0 else 0
    
    hull = cv2.convexHull(largest_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # 7 Hu Invariant Moments
    moments = cv2.moments(largest_cnt)
    hu_moments = cv2.HuMoments(moments).ravel()
    # Log scale transform for Hu moments: -sign(h) * log10(abs(h))
    log_hu = []
    for h in hu_moments:
        if abs(h) > 0:
            log_hu.append(round(float(-1 * np.sign(h) * np.log10(abs(h))), 4))
        else:
            log_hu.append(0.0)
            
    return {
        "Contour_Area": int(area),
        "Perimeter": round(perimeter, 2),
        "Circularity (Compactness)": round(circularity, 4),
        "Aspect_Ratio": round(aspect_ratio, 3),
        "Extent": round(extent, 3),
        "Solidity": round(solidity, 3),
        "Hu_Moment_1": log_hu[0],
        "Hu_Moment_2": log_hu[1],
        "Hu_Moment_3": log_hu[2],
        "Hu_Moment_4": log_hu[3],
        "Hu_Moment_5": log_hu[4],
        "Hu_Moment_6": log_hu[5],
        "Hu_Moment_7": log_hu[6],
    }


def extract_edge_features(img: np.ndarray) -> Dict[str, float]:
    """
    Calculates edge density and gradient energy.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 100, 200)
    
    edge_pixels = np.count_nonzero(edges)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = (edge_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_energy = float(np.mean(sobelx**2 + sobely**2))
    
    return {
        "Edge_Pixel_Count": int(edge_pixels),
        "Edge_Density_Percent": round(edge_density, 3),
        "Mean_Gradient_Energy": round(grad_energy, 2)
    }


def extract_all_features(img: np.ndarray) -> pd.DataFrame:
    """
    Combines all extracted IPCV feature vectors into a clean summary DataFrame.
    """
    color_feats = extract_color_features(img)
    glcm_feats = extract_glcm_texture_features(img)
    shape_feats = extract_shape_features(img)
    edge_feats = extract_edge_features(img)
    
    all_records = []
    
    for k, v in color_feats.items():
        all_records.append({"Feature Category": "Color / Statistical", "Feature Name": k, "Value": str(v)})
    for k, v in glcm_feats.items():
        all_records.append({"Feature Category": "Texture (GLCM)", "Feature Name": k, "Value": str(v)})
    for k, v in shape_feats.items():
        all_records.append({"Feature Category": "Shape / Geometry", "Feature Name": k, "Value": str(v)})
    for k, v in edge_feats.items():
        all_records.append({"Feature Category": "Edge / Boundary", "Feature Name": k, "Value": str(v)})
        
    return pd.DataFrame(all_records)
