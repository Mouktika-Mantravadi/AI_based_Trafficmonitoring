"""
Adaptive Traffic Signal Controller Module.
Converts traffic density categories (LOW, MEDIUM, HIGH) into dynamic adaptive green signal durations
and manages the signal state sequence (GREEN -> YELLOW -> RED).
"""

from typing import Dict, Any, Tuple, Optional
import time
import cv2
import numpy as np
import config


class SignalController:
    """
    Computes dynamic signal green light durations based on traffic density levels
    and manages the traffic signal cycle sequence.
    """

    def __init__(
        self,
        signal_durations: Dict[str, int] = None,
        yellow_duration: int = None
    ):
        """
        :param signal_durations: Custom map for density category -> green time in seconds.
        :param yellow_duration: Yellow transition signal duration in seconds.
        """
        self.signal_durations = signal_durations if signal_durations is not None else config.SIGNAL_DURATIONS
        self.yellow_duration = yellow_duration if yellow_duration is not None else config.YELLOW_DURATION

        # Signal state sequence management
        self.current_state = "GREEN"
        self.state_start_time = time.time()

    def get_green_duration(self, density_category: str) -> int:
        """
        Get recommended green light duration for a given traffic density category.

        :param density_category: 'LOW', 'MEDIUM', or 'HIGH'
        :return: Green duration in seconds
        """
        category = density_category.upper() if density_category else "LOW"
        return self.signal_durations.get(category, 15)

    def compute_signal_timing(self, density_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate dynamic signal green time and signal decision payload based on density results.

        :param density_data: Dictionary output from DensityCalculator
        :return: Signal timing decision dictionary
        """
        category = density_data.get("density_category", "LOW").upper()
        weighted_score = density_data.get("weighted_score", 0.0)

        green_time = self.get_green_duration(category)

        return {
            "density_category": category,
            "weighted_score": weighted_score,
            "recommended_green_time": green_time,
            "yellow_time": self.yellow_duration,
            "current_state": self.current_state,
            "signal_durations": self.signal_durations.copy()
        }

    def update_signal_sequence(self, green_duration: int) -> str:
        """
        Advance signal cycle state machine (GREEN -> YELLOW -> RED -> GREEN).

        :param green_duration: Current recommended green duration in seconds
        :return: Active signal state name ('GREEN', 'YELLOW', or 'RED')
        """
        now = time.time()
        elapsed = now - self.state_start_time

        if self.current_state == "GREEN":
            if elapsed >= green_duration:
                self.current_state = "YELLOW"
                self.state_start_time = now
        elif self.current_state == "YELLOW":
            if elapsed >= self.yellow_duration:
                self.current_state = "RED"
                self.state_start_time = now
        elif self.current_state == "RED":
            # For demo sequence, red holds 5 seconds before cycling back to GREEN
            if elapsed >= 5:
                self.current_state = "GREEN"
                self.state_start_time = now

        return self.current_state

    def draw_signal_overlay(self, frame: np.ndarray, signal_data: Dict[str, Any]) -> np.ndarray:
        """
        Draw active traffic light graphics and timing information overlay on frame.

        :param frame: BGR image array from OpenCV
        :param signal_data: Decision dictionary output from compute_signal_timing()
        :return: Annotated BGR frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        category = signal_data["density_category"]
        score = signal_data["weighted_score"]
        green_time = signal_data["recommended_green_time"]
        state = signal_data["current_state"]

        # Panel Box dimensions on Bottom-Right of frame
        panel_w = 340
        panel_h = 160
        x1 = w - panel_w - 15
        y1 = h - panel_h - 15
        x2 = w - 15
        y2 = h - 15

        # Translucent dark background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Header Title
        cv2.putText(annotated, "ADAPTIVE SIGNAL CONTROL", (x1 + 15, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Traffic Light Graphic Box (Vertical 3 Bulbs: Red, Yellow, Green)
        tl_x1 = x1 + 15
        tl_y1 = y1 + 35
        tl_w = 40
        tl_h = 110
        cv2.rectangle(annotated, (tl_x1, tl_y1), (tl_x1 + tl_w, tl_y1 + tl_h), (40, 40, 40), -1)
        cv2.rectangle(annotated, (tl_x1, tl_y1), (tl_x1 + tl_w, tl_y1 + tl_h), (100, 100, 100), 1)

        # Red bulb (top)
        red_color = (0, 0, 255) if state == "RED" else (40, 40, 80)
        cv2.circle(annotated, (tl_x1 + 20, tl_y1 + 20), 12, red_color, -1)

        # Yellow bulb (middle)
        yellow_color = (0, 255, 255) if state == "YELLOW" else (40, 80, 80)
        cv2.circle(annotated, (tl_x1 + 20, tl_y1 + 55), 12, yellow_color, -1)

        # Green bulb (bottom)
        green_color = (0, 255, 0) if state == "GREEN" else (40, 80, 40)
        cv2.circle(annotated, (tl_x1 + 20, tl_y1 + 90), 12, green_color, -1)

        # Text Metrics next to Traffic Light
        info_x = tl_x1 + tl_w + 15

        cv2.putText(annotated, f"Density Level : {category}", (info_x, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"PCU Score     : {score:.1f}", (info_x, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Highlight Recommended Green Time
        green_text = f"GREEN TIME : {green_time} SEC"
        cv2.putText(annotated, green_text, (info_x, y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Current Signal State Badge
        state_text = f"SIGNAL STATE: {state}"
        state_color = (0, 255, 0) if state == "GREEN" else ((0, 255, 255) if state == "YELLOW" else (0, 0, 255))
        cv2.putText(annotated, state_text, (info_x, y1 + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

        return annotated
