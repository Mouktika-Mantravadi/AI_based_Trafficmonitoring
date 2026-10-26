"""
Complete Smart Traffic Monitoring System.
Pipeline: Detection -> Tracking -> Line Counting -> Density Calculation -> Adaptive Signal Controller.

Usage:
  python run_detection.py --source data/sample_videos/sample_traffic.mp4
  python run_detection.py --source 0
  python run_detection.py --source video.mp4 --line 0.55 --save
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

import config
from src.tracker import VehicleTracker
from src.counter import VehicleCounter
from src.density_calculator import DensityCalculator
from src.signal_controller import SignalController
from src.video_processor import VideoProcessor


def run_full_traffic_monitoring(
    source: str | int,
    conf_threshold: float = 0.4,
    line_position: float = 0.55,
    save_output: bool = False,
    display: bool = True
):
    """
    Complete Traffic Monitoring & Adaptive Signal Timing Pipeline.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    processor = VideoProcessor(source=source)
    if not processor.open():
        print(f"[ERROR] Could not open video source: {source}")
        return

    fps = processor.get_fps()
    width, height = processor.get_dimensions()
    total_frames = processor.get_frame_count()

    print("==================================================")
    print(" AI Traffic Control - Detection, Density & Signal ")
    print("==================================================")
    print(f" Source          : {source}")
    print(f" Resolution      : {width}x{height}")
    print(f" FPS             : {fps:.2f}")
    print(f" Total Frames    : {total_frames if total_frames > 0 else 'Live Stream'}")
    print(f" Target Classes  : {list(config.TARGET_VEHICLE_CLASSES.values())}")
    print(f" Conf Threshold  : {conf_threshold}")
    print(f" Counting Line Y : {int(height * line_position)}px ({int(line_position * 100)}%)")
    print(f" PCU Weights     : {config.VEHICLE_WEIGHTS}")
    print(f" Density Thresh  : {config.DENSITY_THRESHOLDS}")
    print(f" Signal Durations: {config.SIGNAL_DURATIONS}")
    print("==================================================")
    if display:
        print(" Press 'q' in the video window to stop.")
        print("==================================================")

    # Initialize Modules
    tracker = VehicleTracker(conf_threshold=conf_threshold)
    counter = VehicleCounter(line_position=line_position)
    density_calculator = DensityCalculator()
    signal_controller = SignalController()

    # VideoWriter setup if requested
    writer = None
    if save_output:
        output_dir = config.DATA_DIR / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "adaptive_signal_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving output video to: {output_path}")

    frame_index = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = processor.read_frame()
            if not ret or frame is None:
                print("\n[INFO] End of video stream reached.")
                break

            frame_index += 1

            # 1. Multi-object tracking
            tracks = tracker.update(frame, conf_threshold=conf_threshold)

            # 2. Line crossing counter
            counter.update(tracks, frame_height=height, frame_width=width)

            # 3. Active frame vehicle counts breakdown
            active_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for obj in tracks:
                cname = obj["class_name"]
                if cname in active_counts:
                    active_counts[cname] += 1

            # 4. Traffic Density Calculation
            density_data = density_calculator.calculate_density(active_counts)

            # 5. Adaptive Signal Timing Decision
            signal_data = signal_controller.compute_signal_timing(density_data)

            # Update sequence state machine
            active_state = signal_controller.update_signal_sequence(signal_data["recommended_green_time"])
            signal_data["current_state"] = active_state

            # 6. Draw track bounding boxes
            annotated_frame = tracker.draw_tracks(frame, tracks)

            # 7. Draw counter HUD overlay
            annotated_frame = counter.draw_counter_overlay(annotated_frame)

            # 8. Draw Density Status Panel overlay
            annotated_frame = density_calculator.draw_density_overlay(annotated_frame, density_data)

            # 9. Draw Adaptive Signal Light Overlay
            annotated_frame = signal_controller.draw_signal_overlay(annotated_frame, signal_data)

            # Write to output file if enabled
            if writer:
                writer.write(annotated_frame)

            # Display video window if enabled
            if display:
                cv2.imshow("AI Smart Traffic - Adaptive Signal Control", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Run stopped by user.")
                    break

            if frame_index % 30 == 0:
                print(
                    f" Frame {frame_index}/{total_frames if total_frames > 0 else '?'} | "
                    f"Active: {density_data['total_vehicles']} | "
                    f"Score: {density_data['weighted_score']} PCU | "
                    f"Density: {density_data['density_category']} | "
                    f"Green Time: {signal_data['recommended_green_time']}s | "
                    f"State: {signal_data['current_state']}"
                )

    finally:
        elapsed_time = time.time() - start_time
        processor.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        # Cumulative summary
        cumulative_density = density_calculator.calculate_density(counter.counts)
        cumulative_signal = signal_controller.compute_signal_timing(cumulative_density)

        print("\n==================================================")
        print(" Traffic Monitoring & Adaptive Signal Analysis Complete")
        print(f" Processed Frames        : {frame_index}")
        print(f" Total Time               : {elapsed_time:.2f} seconds")
        if elapsed_time > 0:
            print(f" Average FPS              : {frame_index / elapsed_time:.2f}")
        print("--------------------------------------------------")
        print(" CUMULATIVE TRAFFIC METRICS & SIGNAL DECISION:")
        print(f"  • Cars                  : {counter.counts['car']}")
        print(f"  • Motorcycles/Bikes     : {counter.counts['motorcycle']}")
        print(f"  • Buses                 : {counter.counts['bus']}")
        print(f"  • Trucks                : {counter.counts['truck']}")
        print(f"  • TOTAL VEHICLES        : {counter.total_count}")
        print(f"  • WEIGHTED PCU SCORE    : {cumulative_density['weighted_score']} PCU")
        print(f"  • TRAFFIC DENSITY       : {cumulative_density['density_category']}")
        print(f"  • RECOMMENDED GREEN TIME: {cumulative_signal['recommended_green_time']} SECONDS")
        print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Run Smart Traffic Monitoring & Adaptive Signal Timing Module")
    parser.add_argument(
        "--source",
        type=str,
        default=str(config.SAMPLE_VIDEOS_DIR / "sample_traffic.mp4"),
        help="Path to video file or webcam index"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Confidence threshold for YOLO detection"
    )
    parser.add_argument(
        "--line",
        type=float,
        default=0.55,
        help="Counting line position ratio (default: 0.55)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output video to data/output/adaptive_signal_output.mp4"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI window preview"
    )

    args = parser.parse_args()
    run_full_traffic_monitoring(
        source=args.source,
        conf_threshold=args.conf,
        line_position=args.line,
        save_output=args.save,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
