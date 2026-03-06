import os
import time
from datetime import datetime
import cv2
import numpy as np
import yaml
import imageio.v2 as imageio

dll_dir = os.getenv("CPP_POSE_DLL_DIR")
if dll_dir and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(dll_dir)

import cpp_pose
from src.stream.mjpeg import mjpeg_frames
from src.vision.undistort import Undistorter
from src.vision.apriltag_tracker import AprilTagTracker
from src.utils.logger import JsonlLogger, LogRow
from src.control.car_follow import CarFollower


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def ema(prev, x, a):
    return x if prev is None else (1.0 - a) * prev + a * x


def main():
    URL = os.getenv("TAG_STREAM_URL", "http://192.168.4.1:81/stream")
    CAR_IP = os.getenv("CAR_IP", "192.168.4.1")
    CAR_PORT = int(os.getenv("CAR_PORT", "100"))
    CALIB = os.getenv("CAMERA_CALIB_PATH", os.path.join("configs", "camera.yaml"))

    # --- Tag size (side length) ---
    TAG_SIZE_CM = float(os.getenv("TAG_SIZE_CM", "8.0"))
    TAG_SIZE_M = TAG_SIZE_CM / 100.0

    # --- Follow controller parameters ---
    Z_TARGET = 0.3          # meters: desired distance to tag
    Z_DEADBAND = 0.05        # meters: don't move if within +/- this of target
    MAX_SPEED = 0.60         # normalized [0..1] max speed command
    MAX_STEER = 1.00         # normalized [-1..1]

    KX = 1.6                 # steering gain from lateral offset x (m)
    KZ = 1.2                 # speed gain from distance error (m)

    MARGIN_THRESH = 15.0     # ignore weak detections
    LOST_TIMEOUT_S = 0.35    # after this, stop

    # smoothing (helps jitter)
    ALPHA_XZ = 0.20          # smoothing for x,z estimates
    ALPHA_CMD = 0.25         # smoothing for speed/steer commands

    car = CarFollower(
        ip=CAR_IP,
        port=CAR_PORT,
        lost_scan_after_s=5.0,
        z_target=Z_TARGET,
        z_deadband=Z_DEADBAND,
        margin_thresh=MARGIN_THRESH,
    )

    # Load calibration
    with open(CALIB, "r") as f:
        cfg = yaml.safe_load(f)
    K = np.array(cfg["camera_matrix"], dtype=np.float64)
    dist = np.array(cfg["dist_coeffs"], dtype=np.float64)

    # 3D tag corner model (meters)
    half = TAG_SIZE_M / 2.0
    obj_pts = np.array([
        [-half, -half, 0.0],
        [ half, -half, 0.0],
        [ half,  half, 0.0],
        [-half,  half, 0.0],
    ], dtype=np.float64)

    undistorter = Undistorter(CALIB)   # view-only toggle
    tracker = AprilTagTracker(families="tag36h11")

    use_undistort = False  # view only

    # Logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    logger = JsonlLogger(log_dir)
    print(f"[log] writing to {log_dir}")

    # Record MP4 (reliable via imageio-ffmpeg)
    RECORD_MP4 = True
    mp4_w = None
    mp4_h = None
    mp4_path = os.path.join(log_dir, "video.mp4")
    mp4_writer = None  # init after first frame
    mp4_fps = 20
    if RECORD_MP4:
        print(f"[log] recording mp4 to: {mp4_path}")

    # State
    fps_ema = 0.0
    frame_idx = 0
    t0 = time.time()

    last_good_t = None
    x_f = None
    z_f = None
    steer_cmd = None
    speed_cmd = None

    print("Follow-the-tag mode ON.")
    print("Keys: q=quit, u=toggle undistort (view only)")

    try:
        for raw in mjpeg_frames(URL):
            t_start = time.time()
            frame_idx += 1

            # Optional undistort for viewing (PnP uses K+dist, so not needed)
            t_u0 = time.time()
            frame = undistorter(raw) if use_undistort else raw
            undistort_ms = (time.time() - t_u0) * 1000.0

            h, w = frame.shape[:2]
            cx_img = w / 2.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            t_d0 = time.time()
            dets = tracker.detect(gray)
            det = tracker.best_detection(dets)
            detect_ms = (time.time() - t_d0) * 1000.0

            mode_txt = "UNDISTORT: ON (u)" if use_undistort else "UNDISTORT: OFF (u)"
            cv2.putText(frame, mode_txt, (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.line(frame, (int(cx_img), 0), (int(cx_img), h), (255, 255, 255), 1)

            # Defaults for logging (existing fields)
            tag_id = None
            margin = None
            cx_log = None
            cy_log = None
            steer_log = None

            # Pose + follow outputs
            x_m = None
            z_m = None
            yaw_deg = None

            good = False
            now = time.time()

            if det is not None and float(det.decision_margin) >= MARGIN_THRESH:
                good = True
                last_good_t = now

                (cx, cy), corners = tracker.center_and_corners(det)

                # Draw tag outline
                corners_i = corners.astype(int)
                for i in range(4):
                    p1 = tuple(corners_i[i])
                    p2 = tuple(corners_i[(i + 1) % 4])
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
                ok_pnp, x_m, z_m, yaw_deg_cpp, reproj_err = cpp_pose.estimate_pose_square(
                            corners.astype(np.float64),
                            K.astype(np.float64),
                            dist.astype(np.float64),
                            TAG_SIZE_M)
                
                if ok_pnp:
                    # Smooth pose for control stability
                    x_f = ema(x_f, float(x_m), ALPHA_XZ)
                    z_f = ema(z_f, float(z_m), ALPHA_XZ)

                    # send pose to car controller (uses your confidence margin)
                    car.update(x_m=x_f, z_m=z_f, margin=float(det.decision_margin))

                    yaw_deg = float(yaw_deg_cpp)

                    cv2.putText(
                        frame,
                        f"x={x_f:+.2f}m  z={z_f:.2f}m  yaw={yaw_deg:+.1f}deg  err={float(reproj_err):.2f}px",
                        (12, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    # ---- FOLLOW CONTROL ----
                    # Steering: turn toward tag center (x -> 0)
                    steer_raw = clamp(KX * x_f, -MAX_STEER, MAX_STEER)

                    # Speed: keep distance z near target
                    z_err = (z_f - Z_TARGET)
                    if abs(z_err) < Z_DEADBAND:
                        speed_raw = 0.0
                    else:
                        # if too far (z>Z_TARGET) -> positive speed; too close -> negative speed
                        speed_raw = clamp(KZ * z_err, -MAX_SPEED, MAX_SPEED)

                    # Smooth commands to reduce jitter
                    steer_cmd = ema(steer_cmd, steer_raw, ALPHA_CMD)
                    speed_cmd = ema(speed_cmd, speed_raw, ALPHA_CMD)

                    cv2.putText(
                        frame,
                        f"FOLLOW: steer={steer_cmd:+.2f} speed={speed_cmd:+.2f} (target z={Z_TARGET:.2f}m)",
                        (12, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    # For existing log schema, put steering in steer_log
                    steer_log = float(steer_cmd)

                else:
                    good = False  # treat as not good if PnP failed

                # Top text
                cv2.putText(
                    frame,
                    f"id={det.tag_id} margin={det.decision_margin:.1f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                tag_id = int(det.tag_id)
                margin = float(det.decision_margin)
                cx_log = float(cx)
                cy_log = float(cy)

            # LOST behavior (stop)
            lost = (last_good_t is None) or ((now - last_good_t) > LOST_TIMEOUT_S)
            if lost:
                steer_cmd = ema(steer_cmd, 0.0, 0.35)
                speed_cmd = ema(speed_cmd, 0.0, 0.35)
                cv2.putText(frame, "LOST: stopping", (12, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                steer_log = float(steer_cmd) if steer_cmd is not None else None

            # Timing + fps
            total_ms = (time.time() - t_start) * 1000.0
            inst_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)

            if frame_idx % 30 == 0:
                t1 = time.time()
                fps = 30.0 / (t1 - t0)
                t0 = t1
                cv2.setWindowTitle("apriltag_tracking", f"apriltag_tracking | FPS ~ {fps:.1f}")

            # Record MP4 frame (after overlays)
            # Record MP4 frame (after overlays)
            if RECORD_MP4:
                if mp4_writer is None:
                    mp4_h, mp4_w = frame.shape[:2]
                    mp4_writer = imageio.get_writer(mp4_path, fps=mp4_fps, codec="libx264", quality=8)
                if frame.shape[0] != mp4_h or frame.shape[1] != mp4_w:
                    frame_to_write = cv2.resize(frame, (mp4_w, mp4_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_to_write = frame
                mp4_writer.append_data(cv2.cvtColor(frame_to_write, cv2.COLOR_BGR2RGB))

            # Log row (pose fields rely on your updated LogRow dataclass)
            logger.write(LogRow(
                t=time.time(),
                frame_idx=frame_idx,
                w=w, h=h,
                decode_ms=0.0,
                undistort_ms=float(undistort_ms),
                detect_ms=float(detect_ms),
                total_ms=float(total_ms),
                fps_ema=float(fps_ema),
                tag_id=tag_id,
                decision_margin=margin,
                cx=cx_log,
                cy=cy_log,
                steer=steer_log,
                undistort_on=bool(use_undistort),
                x_m=x_f if x_f is not None else x_m,
                z_m=z_f if z_f is not None else z_m,
                yaw_deg=yaw_deg,
            ))
            cv2.putText(
                frame,
                f"FPS ~ {fps_ema:.1f}",
                (12, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2)
            

            cv2.imshow("apriltag_tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u"):
                use_undistort = not use_undistort

    finally:
        if mp4_writer is not None:
            mp4_writer.close()
        logger.close()
        car.close()
        cv2.destroyAllWindows()
        print("[log] closed")


if __name__ == "__main__":
    main()
