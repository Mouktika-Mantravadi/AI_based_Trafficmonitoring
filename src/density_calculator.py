"""
Traffic Density Calculator Module.
Calculates weighted traffic scores based on Passenger Car Units (PCU) and categorizes density levels.
"""

from typing import Dict, Any, Union, List
import cv2
import numpy as np
import config


class DensityCalculator:
    """
    Computes weighted traffic PCU scores and classifies traffic density into LOW, MEDIUM, or HIGH.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        thresholds: Dict[str, float] = None
    ):
        """
        :param weights: Dictionary mapping vehicle category names to PCU weight values.
        :param thresholds: Dictionary specifying thresholds for 'MEDIUM' and 'HIGH' traffic density.
        """
        self.weights = weights if weights is not None else config.VEHICLE_WEIGHTS
        self.thresholds = thresholds if thresholds is not None else config.DENSITY_THRESHOLDS
        self.density_colors = config.DENSITY_COLORS

    def calculate_density(self, vehicle_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate weighted traffic score and categorize density status.

        :param vehicle_counts: Dictionary of counts for each vehicle class (e.g. {'car': 5, 'bus': 2, ...})
        :return: Dictionary containing breakdown, total vehicles, weighted score, and density category.
        """
        total_vehicles = sum(vehicle_counts.values())

        # Calculate weighted PCU score
        weighted_score = 0.0
        for v_type, count in vehicle_counts.items():
            weight = self.weights.get(v_type, 1.0)
            weighted_score += count * weight

        # Categorize traffic density based on configurable thresholds
        medium_thresh = self.thresholds.get("MEDIUM", 10.0)
        high_thresh = self.thresholds.get("HIGH", 25.0)

        if weighted_score >= high_thresh:
            density_category = "HIGH"
        elif weighted_score >= medium_thresh:
            density_category = "MEDIUM"
        else:
            density_category = "LOW"

        return {
            "counts": vehicle_counts.copy(),
            "total_vehicles": total_vehicles,
            "weighted_score": round(weighted_score, 2),
            "density_category": density_category,
            "weights": self.weights.copy(),
            "thresholds": self.thresholds.copy()
        }

    def draw_density_overlay(self, frame: np.ndarray, density_data: Dict[str, Any]) -> np.ndarray:
        """
        Draw traffic density status panel overlay on frame.

        :param frame: BGR image array from OpenCV
        :param density_data: Output dictionary from calculate_density()
        :return: Annotated BGR frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        category = density_data["density_category"]
        score = density_data["weighted_score"]
        total = density_data["total_vehicles"]
        counts = density_data["counts"]

        color = self.density_colors.get(category, (0, 255, 0))

        # Position box on Top-Right of frame
        panel_w = 320
        panel_h = 140
        x1 = w - panel_w - 15
        y1 = 15
        x2 = w - 15
        y2 = y1 + panel_h

        # Semi-transparent dark background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Border color matching density status
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Title: Traffic Density Status
        cv2.putText(annotated, "DENSITY ANALYSIS", (x1 + 15, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Large Category Status Badge (e.g. LOW / MEDIUM / HIGH)
        cv2.rectangle(annotated, (x1 + 15, y1 + 35), (x2 - 15, y1 + 65), color, -1)
        text_color = (0, 0, 0) if category == "MEDIUM" else (255, 255, 255)
        cv2.putText(
            annotated,
            f"DENSITY: {category}",
            (x1 + 45, y1 + 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2,
            cv2.LINE_AA
        )

        # Weighted Score and Vehicles
        score_txt = f"Weighted PCU Score : {score:.1f}"
        total_txt = f"Active Vehicles    : {total}"

        cv2.putText(annotated, score_txt, (x1 + 15, y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, total_txt, (x1 + 15, y1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated
