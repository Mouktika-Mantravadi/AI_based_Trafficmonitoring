"""
Configuration settings for AI-Based Smart Traffic Monitoring System.
"""

from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# YOLO Detection Settings
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano model for fast inference
CONFIDENCE_THRESHOLD = 0.4

# COCO Vehicle Class IDs (car: 2, motorcycle: 3, bus: 5, truck: 7)
TARGET_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Bounding Box Color Mapping (BGR format for OpenCV)
CLASS_COLORS = {
    2: (0, 255, 0),      # Car - Green
    3: (0, 255, 255),    # Motorcycle - Yellow
    5: (255, 255, 0),    # Bus - Cyan
    7: (0, 165, 255)     # Truck - Orange
}

# Web Dashboard (Streamlit) Configuration
PAGE_TITLE = "Smart Traffic Control Dashboard"
PAGE_ICON = "🚦"

# ==============================================================================
# TRAFFIC DENSITY & PCU (Passenger Car Unit) WEIGHTING CONFIGURATION
# ==============================================================================
# PCU weights convert different vehicle types to equivalent passenger car units
# Standard Traffic Engineering Reference PCUs:
# - Car: 1.0 PCU
# - Motorcycle / Bike: 0.5 PCU
# - Bus: 3.0 PCU
# - Truck: 3.0 PCU
VEHICLE_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.5,
    "bus": 3.0,
    "truck": 3.0
}

# Traffic Density Categorization Thresholds (in PCU Score)
# - Score < MEDIUM_THRESHOLD  => LOW Density
# - MEDIUM_THRESHOLD <= Score < HIGH_THRESHOLD => MEDIUM Density
# - Score >= HIGH_THRESHOLD   => HIGH Density
DENSITY_THRESHOLDS = {
    "MEDIUM": 10.0,
    "HIGH": 25.0
}

# Density Category Color Mappings (BGR for OpenCV)
DENSITY_COLORS = {
    "LOW": (0, 255, 0),        # Green
    "MEDIUM": (0, 215, 255),    # Amber / Orange
    "HIGH": (0, 0, 255)         # Red
}

# ==============================================================================
# ADAPTIVE TRAFFIC SIGNAL TIMING CONFIGURATION
# ==============================================================================
# Green light duration (in seconds) based on density level:
# - LOW Density    => 15 seconds
# - MEDIUM Density => 30 seconds
# - HIGH Density   => 45 seconds
SIGNAL_DURATIONS = {
    "LOW": 15,
    "MEDIUM": 30,
    "HIGH": 45
}

# Standard Yellow Signal Transition Duration (in seconds)
YELLOW_DURATION = 5



