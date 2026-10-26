"""
Main Test Script for Smart Traffic Monitoring System.
Verifies project setup and detector module instantiation.
"""

import config
from src.detector import VehicleDetector
from src.video_processor import VideoProcessor


def main():
    print("==================================================")
    print(" AI-Based Smart Traffic Control System ")
    print(" Step 1: Environment & Module Setup Verification ")
    print("==================================================")
    print(f"[CONFIG] Base Directory: {config.BASE_DIR}")
    print(f"[CONFIG] Target Classes: {config.TARGET_VEHICLE_CLASSES}")

    detector = VehicleDetector()
    print("[STATUS] VehicleDetector initialized successfully.")

    processor = VideoProcessor()
    print("[STATUS] VideoProcessor initialized successfully.")

    print("\n[SUCCESS] Environment setup and core modules verified successfully!")


if __name__ == "__main__":
    main()
