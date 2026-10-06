# Wildlife Species Identifier

An AI-powered wildlife image analysis system combining classical **Image Processing and Computer Vision (IPCV)** algorithms with **Transfer Learning (MobileNetV2)** and **YOLOv8 Animal Localization**.

---

## 1. Project Architecture & Directory Structure

```
IPCV Project/
├── app.py                         # Modern Streamlit Wildlife AI Application (Home & Analyze Image)
├── requirements.txt               # Project dependency declarations
├── data/
│   ├── species_metadata.csv       # Scalable biodiversity taxonomy metadata (iNaturalist animal subset)
│   ├── class_mapping.json         # Base class-index to species mapping
│   ├── trained_class_mapping.json # Trained model class-index mapping
│   └── evaluation_metrics.json    # Real measured test evaluation metrics
├── training/
│   ├── dataset_setup.py           # Dataset split creation (train/val/test)
│   ├── train.py                   # Transfer learning fine-tuning on MobileNetV2
│   └── evaluate.py                # Real held-out test evaluation & confusion matrix calculation
├── models/
│   └── wildlife_classifier.pth    # Saved fine-tuned PyTorch classifier checkpoint
├── modules/
│   ├── acquisition.py             # Image reading, validation, metadata extraction
│   ├── preprocessing.py           # Resizing, color conversions, spatial filtering, sharpening, edge detection
│   ├── enhancement.py             # Global Histogram Equalization, CLAHE, Gamma correction
│   ├── morphological.py           # Erosion, Dilation, Opening, Closing, Morphological Gradient
│   ├── segmentation.py            # Otsu thresholding, global thresholding, contour extraction
│   ├── feature_extraction.py      # Color moments, GLCM Haralick texture, geometric shape, Hu moments
│   ├── classification.py          # Dynamic metadata loading, CNN inference, unknown image handling
│   ├── detection.py               # YOLOv8 animal localization & bounding boxes
│   └── evaluation.py              # Real test metric loading & confusion matrix plotting
├── sample_images/                 # High-quality wildlife specimens (Lion, Elephant, Zebra, Tiger, Bear, etc.)
└── evaluation_results/
    └── confusion_matrix.png       # Generated test confusion matrix heatmap
```

---

## 2. Scalable Biodiversity Taxonomy & Dataset Workflow

### Dataset Origin & Structure
The species classification system is based on scalable animal biodiversity taxonomy inspired by the **iNaturalist 2021** benchmark.

Instead of hardcoding species inside Python dictionaries, the system separates metadata from code:
- **`data/species_metadata.csv`**: Contains structured ecological metadata:
  `species_id`, `common_name`, `scientific_name`, `class`, `order`, `family`, `genus`, `conservation_status`, `habitat`, `diet`.
- **`data/trained_class_mapping.json`**: Dynamically maps neural network output neuron indices to species identifiers.

### Dataset Directory Layout
```
dataset/
├── train/
│   ├── african_elephant/
│   ├── bengal_tiger/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

## 3. Training & Evaluation Pipeline

### Step 1: Setup Dataset Hierarchy
```bash
python training/dataset_setup.py
```

### Step 2: Fine-Tune MobileNetV2 Transfer Learning Model
Trains the classification head using Cross-Entropy loss, Cosine Annealing learning rate schedule, and data augmentations (RandomResizedCrop, Horizontal Flip, Rotation, ColorJitter, ImageNet normalization):
```bash
python training/train.py --epochs 5 --batch_size 16 --lr 0.0003
```
*Output*: Saves trained weights to `models/wildlife_classifier.pth` and exports `data/trained_class_mapping.json`.

### Step 3: Run Real Model Evaluation on Held-out Test Data
```bash
python training/evaluate.py
```
*Output*: Computes genuine measured Accuracy, Precision, Recall, F1-Score, and Confusion Matrix saved to `data/evaluation_metrics.json` and `evaluation_results/confusion_matrix.png`.

---

## 4. Unknown & Out-of-Distribution Image Handling

When non-wildlife or ambiguous images (people, vehicles, landscapes, random objects, T-Rex) are uploaded, the classifier computes Softmax confidence scores. If top confidence falls below the confidence threshold:
- The system flags: **"Species not confidently identified"**
- Displays: *"The image does not strongly match the supported wildlife categories."*
- Presents top predictions as candidate possibilities without claiming confirmed identification.

---

## 5. IPCV Algorithms Implemented

| Domain | Algorithm | Academic IPCV Principle |
| :--- | :--- | :--- |
| **Color Spaces** | RGB $\leftrightarrow$ Grayscale $\leftrightarrow$ HSV $\leftrightarrow$ LAB | Decouples luminance $Y = 0.299R + 0.587G + 0.114B$ from chrominance |
| **Noise Reduction** | Gaussian, Median, Bilateral, Box filter | Attenuates high-frequency Gaussian and salt-and-pepper noise |
| **Enhancement** | Global Histogram Equalization & CLAHE | Cumulative Distribution Function (CDF) linearization on luminance |
| **Sharpening** | Unsharp Masking & Laplacian operator | High-pass spatial gradient enhancement: $I_{sharp} = I + \alpha(I - I_{blur})$ |
| **Edge Detection** | Canny Edge Detector & Sobel | Non-maximum suppression and hysteresis thresholding on gradient magnitude |
| **Morphology** | Erosion, Dilation, Opening, Closing | Mathematical morphology using structuring elements ($A \ominus B$, $A \oplus B$) |
| **Segmentation** | Otsu's Bimodal Variance Minimization | Optimal intra-class variance minimization: $\sigma_w^2(T) = \omega_0\sigma_0^2 + \omega_1\sigma_1^2$ |
| **Texture Analysis**| Gray-Level Co-occurrence Matrix (GLCM) | Haralick Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM |
| **Shape Analysis** | 7 Invariant Hu Moments & Circularity | Scale-, rotation-, and translation-invariant geometric moments |
| **Deep Vision** | MobileNetV2 Transfer Learning + YOLOv8 | Fine-grained species classification and single-stage bounding box localization |

---

## 6. How to Run the Web Application

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.
