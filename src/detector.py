"""
Vehicle Detection Module using Ultralytics YOLO.
Handles model initialization, inference, bounding box extraction, and visualization.
"""

from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import config


class VehicleDetector:
    """
    Handles loading the YOLO model, performing detection, and annotating frames with bounding boxes.
    """

    def __init__(self, model_name: str = config.YOLO_MODEL_NAME, conf_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Initialize YOLO detector with specified model weights and confidence threshold.

        :param model_name: Path or model identifier (e.g., 'yolov8n.pt')
        :param conf_threshold: Minimum confidence score to filter detections
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_classes = config.TARGET_VEHICLE_CLASSES
        self.class_colors = config.CLASS_COLORS
        self.model = None

    def load_model(self):
        """Load YOLO model weights into memory."""
        print(f"[INFO] Loading YOLO model: {self.model_name}...")
        self.model = YOLO(self.model_name)
        print("[INFO] YOLO model loaded successfully.")

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Run vehicle detection on a single video frame.

        :param frame: BGR image array (from OpenCV)
        :param conf_threshold: Optional override for confidence threshold
        :return: List of detection dicts containing bbox [x1, y1, x2, y2], confidence, class_id, class_name.
        """
        if self.model is None:
            self.load_model()

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model(frame, conf=threshold, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self.target_classes[cls_id]
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and class labels with confidence scores on the frame.

        :param frame: BGR image array
        :param detections: List of detection dictionaries from detect()
        :return: Annotated BGR frame
        """
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            class_name = det["class_name"].capitalize()
            conf = det["confidence"]

            # Get color for class
            color = self.class_colors.get(cls_id, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Format label text
            label = f"{class_name}: {int(conf * 100)}%"

            # Calculate text background bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position background rectangle above or inside the bbox
            label_y1 = max(y1 - text_height - 6, 0)
            label_y2 = label_y1 + text_height + 6
            label_x2 = x1 + text_width + 8

            # Draw filled background rectangle for text
            cv2.rectangle(annotated_frame, (x1, label_y1), (label_x2, label_y2), color, -1)

            # Draw text in white or dark depending on color
            text_color = (0, 0, 0) if color in [(0, 255, 255), (255, 255, 0)] else (255, 255, 255)
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 4, label_y2 - baseline - 2),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

        return annotated_frame
