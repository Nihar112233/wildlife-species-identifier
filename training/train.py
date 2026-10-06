"""
Wildlife Species Identifier - Transfer Learning Training Script
Fine-tunes MobileNetV2 on wildlife biodiversity classes and saves model checkpoint.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

os.makedirs(MODELS_DIR, exist_ok=True)


def build_wildlife_mobilenet(num_classes: int, pretrained: bool = True):
    """
    Constructs MobileNetV2 with custom classification head for wildlife species.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # Freeze earlier feature extractor layers for fast transfer learning
    for param in model.features[:-4].parameters():
        param.requires_grad = False
        
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model


def get_data_transforms():
    """
    Data augmentations for training and deterministic resizing for validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_wildlife_classifier(dataset_path: str = DATASET_DIR, epochs: int = 5, batch_size: int = 16, lr: float = 3e-4):
    """
    Executes training loop and exports trained weights + metadata mappings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Wildlife Classifier Training on Device: {device}")
    
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print(f"[!] Error: Dataset not found in {train_dir}. Please run dataset preparation first.")
        return False
        
    train_tf, val_tf = get_data_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf) if os.path.exists(val_dir) else train_dataset
    
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"[*] Training on {num_classes} wildlife classes: {classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = build_wildlife_mobilenet(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    model_save_path = os.path.join(MODELS_DIR, "wildlife_classifier.pth")
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            total_train += labels.size(0)
            
        scheduler.step()
        train_acc = (train_correct / total_train) * 100.0 if total_train > 0 else 0.0
        
        # Validation Pass
        model.eval()
        val_correct, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                total_val += labels.size(0)
                
        val_acc = (val_correct / total_val) * 100.0 if total_val > 0 else 0.0
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/total_train:.4f} - Train Acc: {train_acc:.1f}% - Val Acc: {val_acc:.1f}% ({elapsed:.1f}s)")
        
    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "num_classes": num_classes,
        "architecture": "MobileNetV2"
    }, model_save_path)
    print(f"[+] Saved trained model checkpoint to {model_save_path}")
    
    # Save trained class mapping
    class_map = {}
    # Load existing metadata if present
    meta_path = os.path.join(DATA_DIR, "species_metadata.csv")
    meta_df = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    
    for idx, cname in enumerate(classes):
        # Match class name with metadata
        matched_row = None
        if meta_df is not None:
            clean_search = cname.lower().replace("_", " ")
            for _, r in meta_df.iterrows():
                if clean_search in r["common_name"].lower() or r["common_name"].lower() in clean_search:
                    matched_row = r
                    break
                    
        if matched_row is not None:
            class_map[str(idx)] = {
                "class_index": idx,
                "species_id": matched_row["species_id"],
                "common_name": matched_row["common_name"],
                "scientific_name": matched_row["scientific_name"],
                "family": matched_row["family"],
                "genus": matched_row["genus"],
                "order": matched_row.get("order", "N/A"),
                "class": matched_row.get("class", "Mammalia"),
                "conservation_status": matched_row["conservation_status"],
                "habitat": matched_row["habitat"],
                "diet": matched_row["diet"]
            }
        else:
            class_map[str(idx)] = {
                "class_index": idx,
                "species_id": f"SP_{idx:03d}",
                "common_name": cname.replace("_", " ").title(),
                "scientific_name": f"{cname.capitalize()} sp.",
                "family": "Unknown",
                "genus": cname.split("_")[0].capitalize(),
                "order": "N/A",
                "class": "Mammalia",
                "conservation_status": "Not Evaluated",
                "habitat": "Terrestrial",
                "diet": "Not Available"
            }
            
    mapping_path = os.path.join(DATA_DIR, "trained_class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    print(f"[+] Saved trained class mapping to {mapping_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_wildlife_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
