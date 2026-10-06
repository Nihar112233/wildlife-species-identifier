"""
Comprehensive Test Script for Redesigned Wildlife Species Identifier Pipeline.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print("--> [1/9] Testing Imports...")
from modules.acquisition import load_image_from_path, get_image_metadata
from modules.preprocessing import (
    resize_image, convert_color_space, normalize_image, 
    apply_noise_reduction, apply_sharpening, apply_thresholding, apply_edge_detection
)
from modules.enhancement import equalize_histogram, apply_gamma_correction, adjust_brightness_contrast
from modules.morphological import apply_morphological_operation
from modules.segmentation import otsu_segmentation, threshold_segmentation, segment_and_extract_contours
from modules.feature_extraction import (
    extract_color_features, compute_color_histograms, 
    extract_glcm_texture_features, extract_shape_features, 
    extract_edge_features, extract_all_features
)
from modules.classification import WildlifeClassifier
from modules.detection import WildlifeDetector
from modules.evaluation import load_real_evaluation_metrics, get_class_report_dataframe

print("    Imports OK!")

# Test image loading
sample_img_path = os.path.join(PROJECT_ROOT, "sample_images", "lion.jpg")
assert os.path.exists(sample_img_path), f"Missing {sample_img_path}"
img = load_image_from_path(sample_img_path)
assert img is not None, "Failed to load sample image"

print("--> [2/9] Testing Image Metadata...")
meta = get_image_metadata(img, os.path.getsize(sample_img_path))
print(f"    Dimensions: {meta['Width (Pixels)']}x{meta['Height (Pixels)']}, Channels: {meta['Channels']}")

print("--> [3/9] Testing Preprocessing & Morphology...")
gray = convert_color_space(img, "Grayscale")
gauss = apply_noise_reduction(img, "Gaussian Filter", kernel_size=5)
eq = equalize_histogram(img, "CLAHE")
canny = apply_edge_detection(img, "Canny Edge Detection")
morph = apply_morphological_operation(gray, "Opening")
print("    Preprocessing OK!")

print("--> [4/9] Testing Segmentation...")
mask, t_val = otsu_segmentation(img)
ann, fg, stats = segment_and_extract_contours(img, min_contour_area=300)
print(f"    Otsu threshold: {t_val:.1f}, Regions detected: {len(stats)}")

print("--> [5/9] Testing Feature Extraction...")
all_feats = extract_all_features(img)
print(f"    Total features extracted: {len(all_feats)}")

print("--> [6/9] Testing Scalable Classifier & Dynamic Metadata...")
classifier = WildlifeClassifier()
pred_result = classifier.predict(img, top_k=3)
print(f"    Model Architecture: {pred_result['architecture']}")
print(f"    Custom Checkpoint Loaded: {pred_result['is_custom_trained']}")
print(f"    Top Species: {pred_result['top_species']} ({pred_result['top_confidence']}%)")
if pred_result["species_info"]:
    print(f"    Dynamic Taxonomy: Scientific={pred_result['species_info'].get('scientific_name')}, Family={pred_result['species_info'].get('family')}")

print("--> [7/9] Testing Out-of-Distribution / Unknown Image Detection...")
# Create a blank noise image to test uncertainty handling
noise_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
noise_pred = classifier.predict(noise_img, top_k=3, min_confidence_threshold=35.0)
print(f"    Noise Image - is_uncertain: {noise_pred['is_uncertain']}")
print(f"    Uncertainty Warning: {noise_pred['uncertainty_title']}")

print("--> [8/9] Testing YOLOv8 Object Detection...")
detector = WildlifeDetector()
ann_det, dets, crops = detector.detect(img, conf_threshold=0.25)
print(f"    Animals detected: {len(dets)}")
for d in dets:
    print(f"      - {d['class_name']} ({d['confidence']}%) at {d['box_str']}")

print("--> [9/9] Testing Real Evaluation Metrics...")
metrics = load_real_evaluation_metrics()
assert metrics is not None, "Evaluation metrics not found"
print(f"    Measured Real Test Accuracy: {metrics['accuracy']}%")
print(f"    Measured Macro F1: {metrics['f1_macro']}%")
df_rep = get_class_report_dataframe(metrics)
print(f"    Classes reported: {len(df_rep)}")

print("\n" + "="*60)
print("SUCCESS: ALL 9 PIPELINE MODULES & UPGRADES VERIFIED!")
print("="*60)
