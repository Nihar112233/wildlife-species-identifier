"""
Wildlife Species Identifier - Real Model Evaluation Script
Evaluates trained model checkpoint on held-out test data and exports true performance metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_trained_model(test_dir: str = None):
    """
    Evaluates models/wildlife_classifier.pth on real test dataset.
    """
    model_path = os.path.join(MODELS_DIR, "wildlife_classifier.pth")
    if not os.path.exists(model_path):
        print(f"[!] Error: Trained model not found at {model_path}. Train the model first.")
        return None

    if test_dir is None:
        test_dir = os.path.join(DATASET_DIR, "test")
        if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
            test_dir = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"[!] Error: Test directory {test_dir} is empty.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on {test_dir} using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    # Reconstruct architecture
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    import torch.nn as nn
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_labels = [classes[i] for i in y_true]
    y_pred_labels = [classes[i] for i in y_pred]

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)
    report_dict = classification_report(y_true_labels, y_pred_labels, labels=classes, output_dict=True, zero_division=0)

    metrics_data = {
        "status": "Trained & Evaluated",
        "dataset_path": test_dir,
        "total_test_samples": len(y_true),
        "num_classes": num_classes,
        "classes": classes,
        "accuracy": round(float(acc) * 100, 2),
        "precision_macro": round(float(prec_macro) * 100, 2),
        "recall_macro": round(float(rec_macro) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }

    metrics_file = os.path.join(DATA_DIR, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[+] Exported real evaluation metrics to {metrics_file}")

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Wildlife Species Classification - Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved confusion matrix heatmap to {cm_path}")

    print("\n--- Real Performance Summary ---")
    print(f"Accuracy:  {metrics_data['accuracy']}%")
    print(f"Precision: {metrics_data['precision_macro']}%")
    print(f"Recall:    {metrics_data['recall_macro']}%")
    print(f"F1-Score:  {metrics_data['f1_macro']}%")
    return metrics_data


if __name__ == "__main__":
    evaluate_trained_model()
