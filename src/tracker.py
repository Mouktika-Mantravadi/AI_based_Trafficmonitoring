"""
Vehicle Tracking Module using Ultralytics YOLO Tracking (ByteTrack / BotSORT).
Assigns persistent unique track IDs to detected vehicles across video frames.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import config


class VehicleTracker:
    """
    Handles tracking vehicles across consecutive frames using YOLO persist tracking.
    """

    def __init__(self, model_name: str = config.YOLO_MODEL_NAME, conf_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Initialize vehicle tracker.

        :param model_name: YOLO model file path or name
        :param conf_threshold: Minimum confidence threshold for tracking
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_classes = config.TARGET_VEHICLE_CLASSES
        self.class_colors = config.CLASS_COLORS
        self.model = None

    def load_model(self):
        """Load YOLO model weights into memory."""
        print(f"[INFO] Loading YOLO model for tracking: {self.model_name}...")
        self.model = YOLO(self.model_name)
        print("[INFO] Tracker model loaded successfully.")

    def update(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Run tracking on a frame and return active vehicle tracks.

        :param frame: BGR image array
        :param conf_threshold: Optional confidence override
        :return: List of tracked dicts containing track_id, bbox, confidence, class_id, class_name, center.
        """
        if self.model is None:
            self.load_model()

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Run YOLO with tracking persistence
        results = self.model.track(
            frame,
            conf=threshold,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )[0]

        tracked_objects = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    
                    # Extract track ID if available
                    track_id = int(box.id[0].item()) if box.id is not None else None
                    
                    # Calculate center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    tracked_objects.append({
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": self.target_classes[cls_id],
                        "center": (cx, cy)
                    })

        return tracked_objects

    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes, track IDs, vehicle classes, and center points on frame.

        :param frame: BGR image array
        :param tracks: List of track dicts from update()
        :return: Annotated BGR frame
        """
        annotated = frame.copy()

        for obj in tracks:
            x1, y1, x2, y2 = obj["bbox"]
            cls_id = obj["class_id"]
            class_name = obj["class_name"].capitalize()
            conf = obj["confidence"]
            track_id = obj["track_id"]
            cx, cy = obj["center"]

            color = self.class_colors.get(cls_id, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw centroid point
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            # Format label with ID and Class
            if track_id is not None:
                label = f"ID: {track_id} | {class_name} {int(conf * 100)}%"
            else:
                label = f"{class_name} {int(conf * 100)}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_y1 = max(y1 - th - 6, 0)
            label_y2 = label_y1 + th + 6
            label_x2 = x1 + tw + 8

            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)

            text_color = (0, 0, 0) if color in [(0, 255, 255), (255, 255, 0)] else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y2 - baseline - 2),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

        return annotated
