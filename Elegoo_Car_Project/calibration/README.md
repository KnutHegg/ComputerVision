# Calibration Notes

## Camera calibration setup

- Checkerboard pattern: 7x7 inner corners (8x8 squares)
- Square size: 35 mm
- Used frames: 19
- RMS reprojection error: 1.53 px
- P95 per-view error: 0.48 px

## Tracking system snapshot

- Detector: AprilTag `tag36h11`
- Camera stream: ESP32 OV2640 over Wi-Fi (800x600)
- Logging: JSONL detections/timing plus optional replay frames
- Measured p95 processing latency (laptop): about 10.1 ms/frame
- Detection rate (`decision_margin >= 15`): 41.6% (depends on distance/occlusion)
- Average reacquisition after loss: about 1.35 s (`lost-gap = 0.2 s`)

## Current limitations

- Low/unstable lighting reduces detection stability.
