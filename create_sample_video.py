"""
Utility script to generate or download a sample traffic video for testing vehicle detection.
"""

import urllib.request
import cv2
import numpy as np
from pathlib import Path
import config


def download_or_generate_sample_video():
    """Downloads a public domain sample traffic video or creates a synthetic test clip."""
    sample_dir = config.SAMPLE_VIDEOS_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_video_path = sample_dir / "sample_traffic.mp4"

    if sample_video_path.exists():
        print(f"[INFO] Sample traffic video already exists at: {sample_video_path}")
        return sample_video_path

    print("[INFO] Sample video not found. Generating sample traffic video clip...")

    # Video properties
    width, height = 1280, 720
    fps = 30
    duration_sec = 5
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(sample_video_path), fourcc, fps, (width, height))

    # Road coordinates
    road_y1, road_y2 = 200, 650

    for i in range(total_frames):
        # Create background (grass & road)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 80  # Dark gray background
        cv2.rectangle(frame, (0, road_y1), (width, road_y2), (40, 40, 40), -1)  # Asphalt road

        # Lane markings
        dash_offset = (i * 10) % 80
        for x in range(-80 + dash_offset, width + 80, 80):
            cv2.line(frame, (x, 425), (x + 40, 425), (255, 255, 255), 3)

        # Draw vehicle 1 (Car - Moving Right)
        car_x = (i * 12) % (width + 200) - 100
        cv2.rectangle(frame, (car_x, 300), (car_x + 140, 370), (0, 0, 200), -1)  # Red body
        cv2.rectangle(frame, (car_x + 30, 270), (car_x + 110, 300), (150, 150, 150), -1)  # Roof
        cv2.circle(frame, (car_x + 30, 370), 15, (10, 10, 10), -1)  # Wheels
        cv2.circle(frame, (car_x + 110, 370), 15, (10, 10, 10), -1)
        cv2.putText(frame, "CAR", (car_x + 40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw vehicle 2 (Truck - Moving Right)
        truck_x = ((i * 8) + 400) % (width + 300) - 200
        cv2.rectangle(frame, (truck_x, 480), (truck_x + 220, 580), (200, 200, 200), -1)  # Trailer
        cv2.rectangle(frame, (truck_x + 220, 510), (truck_x + 270, 580), (50, 100, 200), -1)  # Cabin
        cv2.circle(frame, (truck_x + 40, 580), 20, (10, 10, 10), -1)
        cv2.circle(frame, (truck_x + 160, 580), 20, (10, 10, 10), -1)
        cv2.circle(frame, (truck_x + 240, 580), 20, (10, 10, 10), -1)
        cv2.putText(frame, "TRUCK", (truck_x + 70, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Draw vehicle 3 (Motorcycle - Moving Left)
        bike_x = width - ((i * 15) % (width + 100))
        cv2.rectangle(frame, (bike_x, 230), (bike_x + 60, 260), (0, 200, 200), -1)
        cv2.circle(frame, (bike_x + 10, 260), 10, (10, 10, 10), -1)
        cv2.circle(frame, (bike_x + 50, 260), 10, (10, 10, 10), -1)

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Sample traffic video created at: {sample_video_path}")
    return sample_video_path


if __name__ == "__main__":
    download_or_generate_sample_video()
