# AI-Based Smart Traffic Monitoring and Adaptive Signal Control Using Vehicle Classification

A modular Python project for ECE final-year capstone project.

## 📁 Project Structure

```
AI model/
├── venv/                      # Python Virtual Environment
├── requirements.txt           # Project Dependencies
├── config.py                  # System Configuration & Path Constants
├── main.py                    # Environment & Core Module Verification Script
├── app.py                     # Streamlit Dashboard Entry Point
├── .gitignore                 # Git Ignore File
├── README.md                  # Project Documentation
├── data/                      # Data Storage
│   ├── models/                # YOLO Model Weights Storage
│   └── sample_videos/         # Input Test Videos
└── src/                       # Modular Source Code
    ├── __init__.py
    ├── detector.py            # Vehicle Detection Module (Ultralytics YOLO)
    ├── video_processor.py     # Video Stream & Frame Processing (OpenCV)
    ├── tracker.py             # Vehicle Tracking (Placeholder)
    ├── density_calculator.py  # Traffic Density Calculation (Placeholder)
    ├── signal_controller.py   # Adaptive Traffic Signal Control Logic (Placeholder)
    └── esp32_comm.py          # ESP32 Hardware Communication (Placeholder)
```

## 🚀 Environment Setup & Running

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Project Environment

Run the verification test script:
```bash
python main.py
```

### 4. Run Streamlit Dashboard

```bash
streamlit run app.py
```
