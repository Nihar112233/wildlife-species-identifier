"""
=============================================================================
Wildlife Species Identifier
AI-Powered Wildlife Image Analysis using Image Processing and Computer Vision
=============================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Modular IPCV Components
from modules.acquisition import load_image_from_bytes, load_image_from_path, get_image_metadata
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
from modules.evaluation import load_real_evaluation_metrics, get_class_report_dataframe, plot_real_confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Wildlife Species Identifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Wildlife AI Theme & Minimalist Palette
st.markdown("""
    <style>
    :root {
        --primary: #2D6A4F;
        --primary-dark: #1B4332;
        --bg-light: #F8F9FA;
        --surface: #FFFFFF;
        --text-main: #1F2937;
        --text-muted: #6B7280;
        --border: #E5E7EB;
    }
    
    .stApp {
        background-color: var(--bg-light);
        color: var(--text-main);
    }
    
    /* Button Customization */
    .stButton>button {
        background-color: #2D6A4F;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 6px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #1B4332;
        color: white;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        padding: 8px 16px;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        color: #2D6A4F !important;
        border-bottom-color: #2D6A4F !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Cache Model Instances
@st.cache_resource(show_spinner="Loading Wildlife Classifier Model...")
def get_classifier():
    return WildlifeClassifier()

@st.cache_resource(show_spinner="Loading YOLOv8 Object Detection Model...")
def get_detector():
    return WildlifeDetector()

# Sample Images
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

def get_available_samples():
    if not os.path.exists(SAMPLE_DIR):
        return []
    return sorted([f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Initialize Navigation State
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

def sync_nav():
    st.session_state.current_page = st.session_state.sidebar_nav_widget

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "last_analyzed_name" not in st.session_state:
    st.session_state.last_analyzed_name = None
if "results_data" not in st.session_state:
    st.session_state.results_data = {}

# Minimal Sidebar Navigation
with st.sidebar:
    st.markdown("### Navigation")
    page_options = ["Home", "Analyze Image"]
    idx = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    st.radio(
        "Select Page:",
        page_options,
        index=idx,
        key="sidebar_nav_widget",
        on_change=sync_nav
    )
    st.markdown("---")
    st.caption("Wildlife AI Vision Platform\nIPCV Edition")

# =============================================================================
# 1. HOME PAGE
# =============================================================================
if st.session_state.current_page == "Home":
    st.title("Wildlife Species Identifier")
    st.subheader("AI-powered wildlife image analysis using Image Processing and Computer Vision")
    st.markdown("---")
    
    if st.button("Analyze an Image →", type="primary"):
        st.session_state.current_page = "Analyze Image"
        st.rerun()
        
    st.markdown("### What this application does")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Image Processing**\n\nObserve grayscale transforms, spatial filtering, contrast enhancement, and edge detection.")
        st.markdown("**Species Recognition**\n\nAutomated classification with top-3 probability distribution and ecological profiling.")
    with c2:
        st.markdown("**Feature Extraction**\n\nQuantify RGB/HSV color moments, GLCM texture matrices, and shape moments.")
        st.markdown("**Wildlife Detection**\n\nYOLOv8 real-time animal localization with bounding box regression.")

    st.markdown("---")
    st.markdown("### Sample Wildlife Gallery")
    samples = get_available_samples()
    if samples:
        gallery_cols = st.columns(min(6, len(samples)))
        for idx, s_name in enumerate(samples[:6]):
            s_path = os.path.join(SAMPLE_DIR, s_name)
            s_img = load_image_from_path(s_path)
            if s_img is not None:
                with gallery_cols[idx]:
                    st.image(s_img, caption=s_name.replace("_", " ").replace(".jpg", "").title(), use_container_width=True)

# =============================================================================
# 2. ANALYZE IMAGE PAGE
# =============================================================================
elif st.session_state.current_page == "Analyze Image":
    st.title("Analyze Wildlife Image")
    st.markdown("Upload a wildlife image or choose a sample to begin analysis.")
    st.markdown("---")

    # STEP 1: CHOOSE IMAGE
    st.subheader("1. Choose Image")
    
    input_mode = st.radio("Selection Source:", ["Sample Wildlife", "Upload Image"], horizontal=True)

    active_img = None
    active_img_name = ""
    active_size = 0

    if input_mode == "Sample Wildlife":
        samples = get_available_samples()
        if samples:
            selected_file = st.selectbox("Select Specimen:", samples, format_func=lambda x: x.replace("_", " ").replace(".jpg", "").title())
            if selected_file:
                sample_path = os.path.join(SAMPLE_DIR, selected_file)
                active_img = load_image_from_path(sample_path)
                active_img_name = selected_file
                active_size = os.path.getsize(sample_path)
        else:
            st.info("No sample images found in sample_images/ directory.")
    else:
        uploaded_file = st.file_uploader("Upload Image File (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            active_size = uploaded_file.size
            active_img_name = uploaded_file.name
            active_img = load_image_from_bytes(uploaded_file.read())

    if active_img_name != st.session_state.last_analyzed_name:
        st.session_state.analysis_done = False

    # STEP 2: PREVIEW IMAGE
    if active_img is not None:
        st.markdown("---")
        st.subheader("2. Preview Image")
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.image(active_img, caption=f"Selected: {active_img_name}", use_container_width=True)
        with p_col2:
            st.markdown("##### Image Properties")
            meta = get_image_metadata(active_img, active_size)
            st.markdown(f"- **Resolution**: {meta.get('Width (Pixels)', 0)} x {meta.get('Height (Pixels)', 0)}")
            st.markdown(f"- **Channels**: {meta.get('Channels', 0)}")
            st.markdown(f"- **Color Space**: {meta.get('Color Representation', 'RGB')}")
            st.markdown(f"- **File Size**: {meta.get('File Size (KB)', 0)} KB")

    # STEP 3: ANALYZE
    if active_img is not None:
        st.markdown("---")
        st.subheader("3. Analyze")
        trigger_analysis = st.button("Analyze Image", type="primary")

        if trigger_analysis:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown("Processing: Loading Image Matrix & Preprocessing...")
            progress_bar.progress(20)
            time.sleep(0.1)

            status_text.markdown("Processing: Extracting Color, Texture & Shape Descriptors...")
            features_df = extract_all_features(active_img)
            color_stats = extract_color_features(active_img)
            color_hists = compute_color_histograms(active_img, bins=32)
            glcm_stats = extract_glcm_texture_features(active_img)
            shape_stats = extract_shape_features(active_img)
            progress_bar.progress(40)
            time.sleep(0.1)

            status_text.markdown("Processing: Performing Mathematical Segmentation (Otsu & Contours)...")
            otsu_mask, optimal_thresh = otsu_segmentation(active_img)
            ann_cnt, fg_seg, cnt_stats = segment_and_extract_contours(active_img, min_contour_area=300)
            progress_bar.progress(60)
            time.sleep(0.1)

            status_text.markdown("Processing: Running CNN Species Classification...")
            classifier = get_classifier()
            clf_results = classifier.predict(active_img, top_k=5)
            progress_bar.progress(80)
            time.sleep(0.1)

            status_text.markdown("Processing: Executing YOLOv8 Animal Detection...")
            detector = get_detector()
            ann_det, detections, crops = detector.detect(active_img, conf_threshold=0.25, animals_only=True)
            progress_bar.progress(100)

            status_text.success("Analysis Complete!")
            time.sleep(0.2)

            st.session_state.results_data = {
                "clf_results": clf_results,
                "detections": detections,
                "annotated_det": ann_det,
                "crops": crops,
                "otsu_mask": otsu_mask,
                "optimal_thresh": optimal_thresh,
                "ann_cnt": ann_cnt,
                "fg_seg": fg_seg,
                "cnt_stats": cnt_stats,
                "features_df": features_df,
                "color_stats": color_stats,
                "color_hists": color_hists,
                "glcm_stats": glcm_stats,
                "shape_stats": shape_stats
            }
            st.session_state.analysis_done = True
            st.session_state.last_analyzed_name = active_img_name
            st.rerun()

    # STEP 4: RESULTS
    if st.session_state.analysis_done and active_img_name == st.session_state.last_analyzed_name:
        st.markdown("---")
        st.subheader("4. View Results")
        
        res = st.session_state.results_data
        clf = res["clf_results"]

        # --- MAIN RESULTS (IMAGE LEFT, PREDICTION RIGHT) ---
        r_left, r_right = st.columns([1, 1])

        with r_left:
            st.image(active_img, caption="Analyzed Specimen", use_container_width=True)

        with r_right:
            if clf.get("is_uncertain", not clf.get("is_confident", True)):
                st.warning("⚠️ **Species Not Confidently Identified**\n\nThe image did not produce a sufficiently confident match to a supported wildlife species.")
                st.markdown(f"**Closest Candidate (Unconfirmed)**: `{clf['top_species']}` ({clf['top_confidence']:.1f}% Match)")
            else:
                st.markdown(f"### Predicted Species\n\n**{clf['top_species']}**")
                st.markdown(f"**Confidence**: {clf['top_confidence']:.1f}%")
                st.progress(min(1.0, clf["top_confidence"] / 100.0))

            st.markdown("#### Top Model Predictions")
            for p in clf["top_predictions"][:3]:
                st.write(f"**{p['class_name']}**: {p['confidence']:.1f}%")
                st.progress(min(1.0, p["confidence"] / 100.0))

            st.markdown("#### Species Information")
            if clf.get("is_uncertain", not clf.get("is_confident", True)):
                st.info("Not available because no species passed the confidence threshold.")
            else:
                info = clf.get("species_info", {})
                if info:
                    st.markdown(f"- **Scientific Name**: *{info.get('scientific_name', 'Not Available')}*")
                    st.markdown(f"- **Family**: {info.get('family', 'Not Available')}")
                    st.markdown(f"- **Genus**: {info.get('genus', 'Not Available')}")
                    st.markdown(f"- **Conservation Status**: **{info.get('conservation_status', 'Not Evaluated')}**")
                    st.markdown(f"- **Native Habitat**: {info.get('habitat', 'Not Available')}")
                    st.markdown(f"- **Diet**: {info.get('diet', 'Not Available')}")

        st.markdown("---")
        
        # --- TABBED IPCV SECTIONS ---
        tab_proc, tab_seg, tab_feat, tab_det, tab_eval = st.tabs([
            "Image Processing",
            "Segmentation",
            "Visual Features",
            "Wildlife Detection",
            "Model Details & Evaluation"
        ])

        with tab_proc:
            st.markdown("#### Image Processing Operations")
            proc_op = st.selectbox(
                "Select IPCV Operation:",
                [
                    "Grayscale Conversion",
                    "Gaussian Filtering",
                    "Median Filtering",
                    "Histogram Equalization",
                    "CLAHE",
                    "Image Sharpening (Unsharp Masking)",
                    "Image Sharpening (Laplacian)",
                    "Intensity Thresholding (Binary)",
                    "Intensity Thresholding (Adaptive)",
                    "Canny Edge Detection",
                    "Morphological Erosion",
                    "Morphological Dilation"
                ]
            )

            p_ctrl, p_view = st.columns([1, 2])
            processed = active_img.copy()

            with p_ctrl:
                if proc_op == "Grayscale Conversion":
                    processed = convert_color_space(active_img, "Grayscale")
                elif proc_op == "Gaussian Filtering":
                    k = st.slider("Kernel Size (Odd):", 3, 15, 5, step=2)
                    sig = st.slider("Sigma:", 0.5, 4.0, 1.5, step=0.5)
                    processed = apply_noise_reduction(active_img, "Gaussian Filter", kernel_size=k, sigma=sig)
                elif proc_op == "Median Filtering":
                    k = st.slider("Kernel Size (Odd):", 3, 15, 5, step=2)
                    processed = apply_noise_reduction(active_img, "Median Filter", kernel_size=k)
                elif proc_op == "Histogram Equalization":
                    processed = equalize_histogram(active_img, method="Standard Global")
                elif proc_op == "CLAHE":
                    clip = st.slider("Clip Limit:", 1.0, 6.0, 2.5, step=0.5)
                    tile = st.slider("Tile Grid Size:", 4, 16, 8, step=2)
                    processed = equalize_histogram(active_img, method="CLAHE", clip_limit=clip, tile_grid_size=tile)
                elif proc_op == "Image Sharpening (Unsharp Masking)":
                    strength = st.slider("Sharpening Strength:", 0.5, 3.0, 1.5, step=0.2)
                    processed = apply_sharpening(active_img, method="Unsharp Masking", strength=strength)
                elif proc_op == "Image Sharpening (Laplacian)":
                    strength = st.slider("Sharpening Strength:", 0.5, 3.0, 1.0, step=0.2)
                    processed = apply_sharpening(active_img, method="Laplacian Sharpening", strength=strength)
                elif proc_op == "Intensity Thresholding (Binary)":
                    t_val = st.slider("Threshold Value:", 0, 255, 127)
                    processed = apply_thresholding(active_img, "Binary Threshold", thresh_val=t_val)
                elif proc_op == "Intensity Thresholding (Adaptive)":
                    bs = st.slider("Block Size (Odd):", 3, 25, 11, step=2)
                    c = st.slider("Constant C:", 0, 10, 2)
                    processed = apply_thresholding(active_img, "Adaptive Gaussian", block_size=bs, c_constant=c)
                elif proc_op == "Canny Edge Detection":
                    low = st.slider("Low Threshold:", 0, 255, 50)
                    high = st.slider("High Threshold:", 0, 255, 150)
                    processed = apply_edge_detection(active_img, "Canny Edge Detection", low_thresh=low, high_thresh=high)
                elif proc_op == "Morphological Erosion":
                    k = st.slider("Structuring Element Size:", 3, 11, 5, step=2)
                    it = st.slider("Iterations:", 1, 4, 1)
                    processed = apply_morphological_operation(active_img, "Erosion", kernel_size=k, iterations=it)
                elif proc_op == "Morphological Dilation":
                    k = st.slider("Structuring Element Size:", 3, 11, 5, step=2)
                    it = st.slider("Iterations:", 1, 4, 1)
                    processed = apply_morphological_operation(active_img, "Dilation", kernel_size=k, iterations=it)

            with p_view:
                st.image(processed, caption=f"Result: {proc_op}", use_container_width=True)

        with tab_seg:
            st.markdown("#### Mathematical Segmentation")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.image(res["otsu_mask"], caption=f"Otsu Optimal Mask (Threshold = {res['optimal_thresh']:.1f})", use_container_width=True)
            with s_col2:
                st.image(res["ann_cnt"], caption="Detected Contours & Bounding Rectangles", use_container_width=True)
            st.image(res["fg_seg"], caption="Segmented Foreground Subject", use_container_width=True)

        with tab_feat:
            st.markdown("#### Quantitative Visual Feature Extraction")
            st.dataframe(res["features_df"], use_container_width=True)
            
            glcm = res["glcm_stats"]
            shape = res["shape_stats"]
            
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("##### GLCM Metrics")
                for k, v in glcm.items():
                    st.markdown(f"- **{k}**: {v}")
            with g2:
                st.markdown("##### Shape Metrics")
                for k, v in shape.items():
                    st.markdown(f"- **{k}**: {v}")

        with tab_det:
            st.markdown("#### YOLOv8 Animal Detection")
            st.image(res["annotated_det"], caption="YOLOv8 Object Bounding Boxes", use_container_width=True)
            dets = res["detections"]
            if dets:
                st.success(f"Detected **{len(dets)}** animal instance(s).")
                st.table(pd.DataFrame([{
                    "Instance": f"#{d['index']}",
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']}%",
                    "Coordinates": d["box_str"]
                } for d in dets]))
            else:
                st.info("No animals detected above threshold.")
            
            if res["crops"]:
                st.markdown("##### Detected Animal Crops")
                crop_cols = st.columns(min(4, len(res["crops"])))
                for idx, cr in enumerate(res["crops"][:4]):
                    with crop_cols[idx]:
                        st.image(cr, caption=f"Crop #{idx+1}", use_container_width=True)

        with tab_eval:
            st.markdown("#### Model Details & Evaluation")
            st.markdown(f"- **Backbone Architecture**: `{clf['architecture']}`")
            st.markdown(f"- **Inference Device**: `{clf['device']}`")
            st.markdown(f"- **Custom Trained Checkpoint Loaded**: `{'Yes' if clf['is_custom_trained'] else 'No'}`")

            real_metrics = load_real_evaluation_metrics()
            if real_metrics is not None:
                st.markdown("##### Measured Test Benchmark Evaluation")
                em1, em2, em3, em4 = st.columns(4)
                em1.metric("Accuracy", f"{real_metrics['accuracy']}%")
                em2.metric("Macro Precision", f"{real_metrics['precision_macro']}%")
                em3.metric("Macro Recall", f"{real_metrics['recall_macro']}%")
                em4.metric("Macro F1", f"{real_metrics['f1_macro']}%")

                cm_left, cm_right = st.columns([3, 2])
                with cm_left:
                    fig_cm = plot_real_confusion_matrix(real_metrics["confusion_matrix"], real_metrics["classes"])
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
                with cm_right:
                    report_df = get_class_report_dataframe(real_metrics)
                    st.dataframe(report_df, use_container_width=True)
            else:
                st.info("Model not evaluated on test set yet.")
