"""
Module: Model Evaluation & Performance Metrics
Handles loading and visualization of real measured validation/test evaluation metrics.
Displays genuine evaluation data only.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METRICS_FILE = os.path.join(DATA_DIR, "evaluation_metrics.json")


def load_real_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """
    Loads real evaluation results computed on held-out test data.
    Returns None if the model has not been evaluated yet.
    """
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"[!] Error loading evaluation metrics: {e}")
            return None
    return None


def get_class_report_dataframe(metrics_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts real classification report dictionary into a pandas DataFrame.
    """
    if not metrics_data or "classification_report" not in metrics_data:
        return pd.DataFrame()
        
    rep = metrics_data["classification_report"]
    rows = []
    classes = metrics_data.get("classes", [])
    
    for cls_name in classes:
        if cls_name in rep:
            c_dict = rep[cls_name]
            rows.append({
                "Species Class": cls_name.replace("_", " ").title(),
                "Precision (%)": round(c_dict.get("precision", 0) * 100, 2),
                "Recall (%)": round(c_dict.get("recall", 0) * 100, 2),
                "F1-Score (%)": round(c_dict.get("f1-score", 0) * 100, 2),
                "Test Support": int(c_dict.get("support", 0))
            })
            
    return pd.DataFrame(rows)


def plot_real_confusion_matrix(cm_matrix: List[List[int]], classes: List[str]) -> plt.Figure:
    """
    Renders real confusion matrix heatmap.
    """
    cm = np.array(cm_matrix)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(4.5, len(classes) * 0.7)))
    clean_labels = [c.replace("_", "\n").title() for c in classes]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=[c.replace("_", " ").title() for c in classes],
        cbar=True,
        ax=ax
    )
    ax.set_title("Measured Test Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Species", fontsize=9)
    ax.set_ylabel("Ground Truth Species", fontsize=9)
    plt.tight_layout()
    return fig
