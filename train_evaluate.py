import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def build_wildlife_model(num_classes: int = 12, backbone: str = "mobilenet_v2", pretrained: bool = True):
    if backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
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
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_model(data_dir: str, num_epochs: int = 5, batch_size: int = 16, lr: float = 1e-4, backbone: str = "mobilenet_v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    train_tf, val_tf = get_data_transforms()
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    model = build_wildlife_model(num_classes=num_classes, backbone=backbone)
    model.to(device)
    
    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_model_state = None

    print(f"[*] Starting training for {num_epochs} epochs on {num_classes} classes...")
    for epoch in range(num_epochs):
        start_t = time.time()
        
        # Unfreeze backbone after 2 epochs
        if epoch == 2:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr/10)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
        
        val_acc = (val_correct / val_total) * 100.0
        elapsed = time.time() - start_t
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}% ({elapsed:.1f}s)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            
    os.makedirs("models", exist_ok=True)
    save_path = "models/wildlife_classifier.pth"
    torch.save({
        "classes": train_dataset.classes,
        "model_state_dict": best_model_state
    }, save_path)
    print(f"[+] Model checkpoint saved to {save_path}")
    
    evaluate_benchmark(data_dir, save_path)

def evaluate_benchmark(data_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("WILDLIFE SPECIES IDENTIFIER - ACADEMIC BENCHMARK EVALUATION")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    
    model = build_wildlife_model(num_classes=len(classes), backbone="mobilenet_v2")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    _, val_tf = get_data_transforms()
    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision_macro": round(prec_mac * 100, 2),
        "recall_macro": round(rec_mac * 100, 2),
        "f1_macro": round(f1_mac * 100, 2),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    }
    
    print(f"Overall Accuracy:  {metrics['accuracy']}%")
    print(f"Macro Precision:   {metrics['precision_macro']}%")
    print(f"Macro Recall:      {metrics['recall_macro']}%")
    print(f"Macro F1-Score:    {metrics['f1_macro']}%")
    
    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n[+] Metrics saved to data/evaluation_metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Classifier Training & Evaluation")
    parser.add_argument("--mode", choices=["eval", "train"], default="train", help="Execution mode")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    args = parser.parse_args()
    if args.mode == "eval":
        evaluate_benchmark(args.data_dir, "models/wildlife_classifier.pth")
    elif args.mode == "train":
        train_model(args.data_dir, num_epochs=args.epochs)
