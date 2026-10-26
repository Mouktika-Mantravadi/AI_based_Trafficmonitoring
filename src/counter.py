"""
Vehicle Line Counter Module.
Counts tracked vehicles crossing a virtual counting line and maintains separate counts per class.
"""

from typing import List, Dict, Any, Tuple, Set
import cv2
import numpy as np


class VehicleCounter:
    """
    Manages counting lines, line-crossing detection, unique ID tracking, and counting metrics.
    """

    def __init__(self, line_position: float = 0.55, offset: int = 6):
        """
        :param line_position: Y-axis ratio of counting line (e.g., 0.55 = 55% from top of frame)
        :param offset: Pixel tolerance buffer around counting line
        """
        self.line_position = line_position
        self.offset = offset

        # Counting state
        self.counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }
        self.total_count = 0
        self.counted_ids: Set[int] = set()
        self.prev_centers: Dict[int, Tuple[int, int]] = {}

    def update(self, tracks: List[Dict[str, Any]], frame_height: int, frame_width: int):
        """
        Process tracked vehicles for the current frame and check line crossings.

        :param tracks: List of tracked dicts containing track_id, bbox, class_name, center.
        :param frame_height: Video frame height in pixels
        :param frame_width: Video frame width in pixels
        """
        line_y = int(frame_height * self.line_position)

        current_frame_ids = set()

        for obj in tracks:
            track_id = obj.get("track_id")
            if track_id is None:
                continue

            current_frame_ids.add(track_id)
            cx, cy = obj["center"]
            class_name = obj["class_name"]

            # If vehicle was tracked in previous frame, check for line crossing
            if track_id in self.prev_centers:
                prev_cx, prev_cy = self.prev_centers[track_id]

                # Check if vehicle crossed the line (moving downwards or upwards)
                crossed_down = (prev_cy < line_y) and (cy >= line_y)
                crossed_up = (prev_cy > line_y) and (cy <= line_y)
                crossed_near = abs(cy - line_y) <= self.offset

                if (crossed_down or crossed_up or crossed_near) and (track_id not in self.counted_ids):
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    if class_name in self.counts:
                        self.counts[class_name] += 1
                    else:
                        self.counts[class_name] = 1
                    print(f"[COUNT LOG] Vehicle ID #{track_id} ({class_name.upper()}) crossed line! Total: {self.total_count}")

            # Update previous center
            self.prev_centers[track_id] = (cx, cy)

        # Cleanup lost tracks from prev_centers cache to free memory
        for tid in list(self.prev_centers.keys()):
            if tid not in current_frame_ids and tid not in self.counted_ids:
                # Keep counted IDs in prev_centers for a while or clean uncounted lost tracks
                pass

    def draw_counter_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the virtual counting line and live count metrics panel on the frame.

        :param frame: BGR image array
        :return: Annotated BGR frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        line_y = int(h * self.line_position)

        # 1. Draw Counting Line
        line_color = (0, 0, 255)  # Bright Red
        cv2.line(annotated, (0, line_y), (w, line_y), line_color, 3)

        # Label counting line
        cv2.putText(
            annotated,
            "--- COUNTING LINE ---",
            (w - 240, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # 2. Draw Count Metrics Dashboard Overlay (Top-Left translucent box)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (330, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (330, 150), (0, 255, 0), 2)

        # Header
        cv2.putText(annotated, "TRAFFIC COUNTER", (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Counts
        car_txt = f"Cars: {self.counts.get('car', 0)}"
        bike_txt = f"Bikes: {self.counts.get('motorcycle', 0)}"
        bus_txt = f"Buses: {self.counts.get('bus', 0)}"
        truck_txt = f"Trucks: {self.counts.get('truck', 0)}"

        cv2.putText(annotated, car_txt, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, bike_txt, (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, bus_txt, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated, truck_txt, (160, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Total Count
        cv2.line(annotated, (20, 100), (320, 100), (100, 100, 100), 1)
        cv2.putText(annotated, f"TOTAL VEHICLES: {self.total_count}", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def reset(self):
        """Reset all counters and tracked state."""
        self.counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.total_count = 0
        self.counted_ids.clear()
        self.prev_centers.clear()
