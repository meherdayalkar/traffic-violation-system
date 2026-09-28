# detection/detector.py — CV Engineer's module
# Handles YOLOv8 detection: bikes, persons, helmets

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class Detection:
    """Single detection result from YOLO."""
    bbox:       np.ndarray      # [x1, y1, x2, y2]
    class_id:   int
    class_name: str
    confidence: float
    frame_id:   int = 0

    @property
    def center(self):
        return (
            int((self.bbox[0] + self.bbox[2]) / 2),
            int((self.bbox[1] + self.bbox[3]) / 2),
        )

    @property
    def area(self):
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class VehicleDetector:
    """
    CV Engineer module.
    Wraps YOLOv8 for traffic scene detection.
    Outputs clean Detection objects ready for the tracker.
    """

    # Classes we care about — subset of COCO
    VEHICLE_CLASSES  = {"motorcycle", "bicycle", "car", "bus", "truck"}
    PERSON_CLASSES   = {"person"}
    # If you add a custom helmet model, list its class names here:
    HELMET_CLASSES   = {"helmet", "no-helmet", "head"}

    def __init__(self, model_path: str = config.YOLO_MODEL):
        logger.info(f"Loading YOLO model: {model_path} on {config.DEVICE}")
        try:
            self.model = YOLO(model_path)
            self.model.to(config.DEVICE)
        except Exception as e:
            logger.warning(f"GPU load failed ({e}), falling back to CPU")
            self.model = YOLO(model_path)
        self.frame_count = 0
        logger.success("Detector ready")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.
        Returns list of Detection objects.
        """
        self.frame_count += 1
        results = self.model(
            frame,
            conf=config.YOLO_CONFIDENCE,
            iou=config.YOLO_IOU,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id].lower()
            conf     = float(box.conf[0])
            bbox     = box.xyxy[0].cpu().numpy().astype(int)

            # Filter to only relevant classes
            if not self._is_relevant(cls_name):
                continue

            detections.append(Detection(
                bbox       = bbox,
                class_id   = cls_id,
                class_name = cls_name,
                confidence = conf,
                frame_id   = self.frame_count,
            ))

        return detections

    def _is_relevant(self, cls_name: str) -> bool:
        all_relevant = (
            self.VEHICLE_CLASSES |
            self.PERSON_CLASSES  |
            self.HELMET_CLASSES
        )
        return cls_name in all_relevant

    def get_vehicles(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_name in self.VEHICLE_CLASSES]

    def get_persons(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_name in self.PERSON_CLASSES]

    def get_helmets(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_name in self.HELMET_CLASSES]

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame (for debugging)."""
        COLOR_MAP = {
            "motorcycle": (0, 140, 255),
            "bicycle":    (0, 140, 255),
            "person":     (255, 200, 0),
            "helmet":     (0, 220, 0),
            "no-helmet":  (0, 0, 255),
            "head":       (200, 200, 0),
        }
        frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = COLOR_MAP.get(det.class_name, (180, 180, 180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
