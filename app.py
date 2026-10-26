"""
Streamlit Web Interface for Smart Traffic Control Dashboard - Step 5: Adaptive Signal Control.
"""

import tempfile
from pathlib import Path
import cv2
import streamlit as st

import config
from src.tracker import VehicleTracker
from src.counter import VehicleCounter
from src.density_calculator import DensityCalculator
from src.signal_controller import SignalController
from src.video_processor import VideoProcessor

# Configure Streamlit page layout
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide"
)

st.title("🚦 AI-Based Smart Traffic Monitoring - Adaptive Signal Control")
st.caption("Step 5: Software-Based Adaptive Traffic Signal Timing Logic (LOW: 15s | MEDIUM: 30s | HIGH: 45s)")

# Sidebar Configuration
st.sidebar.header("⚙️ Detection & Tracking Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=config.CONFIDENCE_THRESHOLD,
    step=0.05
)

line_position_ratio = st.sidebar.slider(
    "Counting Line Position (Y-Axis %)",
    min_value=0.1,
    max_value=0.9,
    value=0.55,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ PCU Vehicle Weights")
weight_car = st.sidebar.number_input("Car Weight (PCU)", min_value=0.1, max_value=10.0, value=config.VEHICLE_WEIGHTS["car"], step=0.1)
weight_bike = st.sidebar.number_input("Motorcycle / Bike Weight (PCU)", min_value=0.1, max_value=10.0, value=config.VEHICLE_WEIGHTS["motorcycle"], step=0.1)
weight_bus = st.sidebar.number_input("Bus Weight (PCU)", min_value=0.1, max_value=10.0, value=config.VEHICLE_WEIGHTS["bus"], step=0.5)
weight_truck = st.sidebar.number_input("Truck Weight (PCU)", min_value=0.1, max_value=10.0, value=config.VEHICLE_WEIGHTS["truck"], step=0.5)

custom_weights = {
    "car": weight_car,
    "motorcycle": weight_bike,
    "bus": weight_bus,
    "truck": weight_truck
}

st.sidebar.markdown("---")
st.sidebar.header("🚦 Adaptive Signal Timing Settings")

low_green_sec = st.sidebar.number_input("LOW Density Green Time (Sec)", min_value=5, max_value=120, value=config.SIGNAL_DURATIONS["LOW"], step=5)
med_green_sec = st.sidebar.number_input("MEDIUM Density Green Time (Sec)", min_value=5, max_value=120, value=config.SIGNAL_DURATIONS["MEDIUM"], step=5)
high_green_sec = st.sidebar.number_input("HIGH Density Green Time (Sec)", min_value=5, max_value=120, value=config.SIGNAL_DURATIONS["HIGH"], step=5)

custom_signal_durations = {
    "LOW": low_green_sec,
    "MEDIUM": med_green_sec,
    "HIGH": high_green_sec
}

# Input Video Source Selection
st.subheader("📹 Select Traffic Video Source")
source_option = st.radio(
    "Choose video input method:",
    ("Sample Video", "Upload Local Video File", "Specify Local Video Path"),
    horizontal=True
)

video_path = None

if source_option == "Sample Video":
    sample_path = config.SAMPLE_VIDEOS_DIR / "sample_traffic.mp4"
    if sample_path.exists():
        video_path = str(sample_path)
        st.info(f"Using sample video: `{sample_path.name}`")
    else:
        st.warning("Sample video not found. Run `python create_sample_video.py` or upload a video file.")

elif source_option == "Upload Local Video File":
    uploaded_file = st.file_uploader("Upload a traffic video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.success(f"Uploaded file loaded: {uploaded_file.name}")

elif source_option == "Specify Local Video Path":
    path_input = st.text_input("Enter absolute or relative path to video file:")
    if path_input:
        if Path(path_input).exists():
            video_path = path_input
            st.success(f"Valid video file selected: `{path_input}`")
        else:
            st.error("Specified video file does not exist. Please check the file path.")

# Monitoring & Adaptive Control Execution
if video_path:
    if st.button("▶️ Start Adaptive Signal Control", type="primary"):
        st.markdown("---")
        st.subheader("📺 Live Video Stream with Adaptive Signal Control HUD")

        col1, col2 = st.columns([3, 1])

        with col1:
            st_frame = st.empty()

        with col2:
            st.markdown("#### 🚥 Signal Decision Dashboard")
            signal_state_ph = st.empty()
            recommended_green_ph = st.empty()
            density_status_ph = st.empty()
            pcu_score_ph = st.empty()
            
            st.markdown("---")
            st.markdown("#### 📊 Vehicle Breakdown")
            car_metric = st.metric("Cars", "0")
            bike_metric = st.metric("Bikes", "0")
            bus_metric = st.metric("Buses", "0")
            truck_metric = st.metric("Trucks", "0")

        # Initialize pipeline modules
        tracker = VehicleTracker(conf_threshold=conf_threshold)
        counter = VehicleCounter(line_position=line_position_ratio)
        density_calculator = DensityCalculator(weights=custom_weights)
        signal_controller = SignalController(signal_durations=custom_signal_durations)
        processor = VideoProcessor(source=video_path)

        if not processor.open():
            st.error("Failed to open video stream.")
        else:
            stop_btn = st.button("⏹️ Stop Signal Control")
            height = processor.get_dimensions()[1]
            width = processor.get_dimensions()[0]

            while not stop_btn:
                ret, frame = processor.read_frame()
                if not ret or frame is None:
                    st.success("Video processing completed.")
                    break

                # 1. Tracking
                tracks = tracker.update(frame, conf_threshold=conf_threshold)

                # 2. Line Counter
                counter.update(tracks, frame_height=height, frame_width=width)

                # 3. Active counts
                active_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
                for obj in tracks:
                    cname = obj["class_name"]
                    if cname in active_counts:
                        active_counts[cname] += 1

                # 4. Density Calculation
                density_data = density_calculator.calculate_density(active_counts)

                # 5. Adaptive Signal Timing Decision
                signal_data = signal_controller.compute_signal_timing(density_data)
                active_state = signal_controller.update_signal_sequence(signal_data["recommended_green_time"])
                signal_data["current_state"] = active_state

                # 6. Render bounding boxes and HUD overlays
                annotated_frame = tracker.draw_tracks(frame, tracks)
                annotated_frame = counter.draw_counter_overlay(annotated_frame)
                annotated_frame = density_calculator.draw_density_overlay(annotated_frame, density_data)
                annotated_frame = signal_controller.draw_signal_overlay(annotated_frame, signal_data)

                # Render frame in Streamlit UI
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update live Streamlit signal metrics
                curr_state = signal_data["current_state"]
                if curr_state == "GREEN":
                    signal_state_ph.success("### 🟢 SIGNAL: GREEN")
                elif curr_state == "YELLOW":
                    signal_state_ph.warning("### 🟡 SIGNAL: YELLOW")
                else:
                    signal_state_ph.error("### 🔴 SIGNAL: RED")

                recommended_green_ph.metric("Recommended Green Time", f"{signal_data['recommended_green_time']} SECONDS")
                
                cat = density_data["density_category"]
                density_status_ph.info(f"Traffic Density: **{cat}**")
                pcu_score_ph.metric("PCU Score", f"{density_data['weighted_score']:.1f} PCU")

                car_metric.metric("Cars", f"{active_counts['car']} (Counted: {counter.counts['car']})")
                bike_metric.metric("Bikes", f"{active_counts['motorcycle']} (Counted: {counter.counts['motorcycle']})")
                bus_metric.metric("Buses", f"{active_counts['bus']} (Counted: {counter.counts['bus']})")
                truck_metric.metric("Trucks", f"{active_counts['truck']} (Counted: {counter.counts['truck']})")

            processor.release()
