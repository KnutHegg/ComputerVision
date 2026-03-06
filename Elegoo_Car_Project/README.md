# Elegoo Car Vision + AprilTag Tracking

Computer vision and control utilities for an Elegoo robot car using an ESP32 camera stream, camera calibration, AprilTag tracking, and optional autonomous follow behavior.

## What is included

- Calibration utilities to estimate camera intrinsics and distortion
- MJPEG stream capture and logging tools
- AprilTag detection + pose estimation pipeline
- Car control helpers (keyboard drive and follow controller)
- C++ pose extension source (`calibration/cpp_pose_build`) for faster pose estimation

## Repository layout

- `calibration/app/`: runnable apps (`run_tag_tracking.py`, `drive_with_keyboard.py`)
- `calibration/src/`: reusable modules (stream, vision, control, logging)
- `calibration/tools/`: calibration/reporting/recording scripts
- `calibration/configs/`: local config files such as camera intrinsics
- `calibration/README.md`: calibration/tracking performance notes

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run AprilTag tracking:

```powershell
cd calibration
python app/run_tag_tracking.py
```

## Runtime configuration

The apps now read network/runtime settings from environment variables so you do not need to edit source code before sharing.

- `TAG_STREAM_URL` (default: `http://192.168.4.1:81/stream`)
- `CAR_IP` (default: `192.168.4.1`)
- `CAR_PORT` (default: `100`)
- `CAMERA_CALIB_PATH` (default: `configs/camera.yaml`)
- `TAG_SIZE_CM` (default: `8.0`)
- `CPP_POSE_DLL_DIR` (optional, Windows only)

Example (PowerShell):

```powershell
$env:TAG_STREAM_URL = "http://192.168.4.1:81/stream"
$env:CAR_IP = "192.168.4.1"
$env:CAR_PORT = "100"
cd calibration
python app/run_tag_tracking.py
```

## Notes before publishing

- Generated logs, calibration captures, reports, and build outputs are ignored by `.gitignore`.
- The tracked code should be source/config only; large runtime artifacts should stay local.
