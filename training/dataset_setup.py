"""
Training: Dataset Setup & Preparation
Organizes animal wildlife datasets into train, val, and test splits following iNaturalist/biodiversity standards.
"""

import os
import shutil
import random
import argparse
from PIL import Image

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")


def setup_dataset_structure(classes=None, create_sample_splits=True):
    """
    Creates standard dataset/ train, val, test folder hierarchy.
    If sample images exist, populates initial splits to verify pipeline readiness.
    """
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(DATASET_ROOT, s), exist_ok=True)

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_images")
    if create_sample_splits and os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for sf in sample_files:
            class_name = os.path.splitext(sf)[0]
            src_path = os.path.join(sample_dir, sf)
            
            for s in splits:
                cls_dir = os.path.join(DATASET_ROOT, s, class_name)
                os.makedirs(cls_dir, exist_ok=True)
                dest_path = os.path.join(cls_dir, f"{class_name}_01.jpg")
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)
                    
        print(f"[+] Populated dataset hierarchy across {len(sample_files)} sample classes in {DATASET_ROOT}")


if __name__ == "__main__":
    setup_dataset_structure()
