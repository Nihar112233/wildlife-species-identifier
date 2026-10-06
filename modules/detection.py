"""
Module: Wildlife Object Detection
Implements real-time animal and wildlife object detection using pretrained YOLOv8
with bounding box annotation, confidence filtering, and region crop extraction.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


# COCO Animal Class IDs
COCO_ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe"
}


class WildlifeDetector:
    """
    Manages YOLOv8 object detection model for wildlife detection.
    """
    def __init__(self, model_weight: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_weight)

    def detect(self, img_rgb: np.ndarray, conf_threshold: float = 0.25, 
               animals_only: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]], List[np.ndarray]]:
        """
        Executes YOLO object detection on the RGB image.
        
        Args:
            img_rgb: Input RGB image numpy array.
            conf_threshold: Minimum confidence threshold.
            animals_only: If True, filters only animal classes.
            
        Returns:
            annotated_img: RGB image with drawn bounding boxes and labels.
            detections: List of detection dictionaries.
            cropped_objects: List of cropped sub-images.
        """
        # YOLO expects RGB or BGR numpy array
        results = self.model(img_rgb, conf=conf_threshold, verbose=False)
        result = results[0]
        
        annotated_img = img_rgb.copy()
        detections = []
        cropped_objects = []
        
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return annotated_img, [], []
            
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            
            # Filter animal classes if requested
            if animals_only and cls_id not in COCO_ANIMAL_CLASSES:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Bound coordinates within image dimensions
            h, w = img_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop object patch
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                cropped_objects.append(crop)
                
            # Draw bounding box
            color = (0, 255, 127) if cls_id in COCO_ANIMAL_CLASSES else (255, 165, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label banner
            label_str = f"{class_name.capitalize()} {conf*100:.1f}%"
            (label_w, label_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - label_h - 10)), (x1 + label_w + 6, max(0, y1)), color, -1)
            cv2.putText(annotated_img, label_str, (x1 + 3, max(label_h + 2, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            detections.append({
                "index": i + 1,
                "class_name": class_name.capitalize(),
                "is_animal": cls_id in COCO_ANIMAL_CLASSES,
                "confidence": round(conf * 100, 2),
                "box": (x1, y1, x2, y2),
                "box_str": f"[{x1}, {y1}, {x2}, {y2}]",
                "area_px": (x2 - x1) * (y2 - y1)
            })
            
        return annotated_img, detections, cropped_objects
