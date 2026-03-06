import argparse
import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO

from common import CLASS_NAMES, compute_minimap_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live LoL minimap inference overlay.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/minimap_champion_detector/weights/best.pt"),
        help="Path to trained model.",
    )
    parser.add_argument("--window-title", type=str, default="League of Legends (TM) Client")
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--minimap-w", type=int, default=300)
    parser.add_argument("--minimap-h", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=0.82)
    parser.add_argument("--contrast", type=float, default=0.77)
    parser.add_argument("--brightness", type=int, default=-15)
    parser.add_argument("--left-offset", type=int, default=32)
    parser.add_argument("--top-offset", type=int, default=32)
    parser.add_argument("--right-offset", type=int, default=16)
    parser.add_argument("--bottom-offset", type=int, default=11)
    return parser.parse_args()


def build_gamma_table(gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.float32)
    return table.astype("uint8")


def get_game_window(window_title: str):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise RuntimeError(f"Game window not found: {window_title}")
    return windows[0]


def get_capture_region(game_window) -> tuple[int, int, int, int]:
    left = game_window.left
    top = game_window.top
    right = left + game_window.width
    bottom = top + game_window.height
    return left, top, right, bottom


def get_box_color(class_name: str) -> tuple[int, int, int]:
    if "Hecarim" in class_name:
        return 255, 255, 255
    if class_name.startswith("ally_"):
        return 0, 255, 0
    if class_name.startswith("enemy_"):
        return 0, 0, 255
    return 255, 255, 255


def draw_detections(minimap: np.ndarray, results, class_names: list[str]) -> np.ndarray:
    annotated = minimap.copy()
    if not results or results[0].boxes is None:
        return annotated

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = get_box_color(class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        label = f"{class_name} {conf:.2f}"
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_top = max(0, y1 - text_h - baseline)
        text_bottom = max(0, y1)
        cv2.rectangle(annotated, (x1, text_top), (x1 + text_w, text_bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, max(text_h, y1 - baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
    return annotated


def setup_windows(game_window, minimap_w: int, minimap_h: int) -> None:
    cv2.namedWindow("Full Game + Minimap Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Full Game + Minimap Overlay", game_window.width, game_window.height)
    cv2.namedWindow("Minimap Only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Minimap Only", minimap_w * 2, minimap_h * 2)
    cv2.moveWindow("Full Game + Minimap Overlay", game_window.left, game_window.top)
    cv2.moveWindow("Minimap Only", game_window.left + game_window.width + 10, game_window.top)


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return

    gamma_table = build_gamma_table(args.gamma)
    model = YOLO(str(args.model))

    try:
        game_window = get_game_window(args.window_title)
    except RuntimeError as error:
        print(error)
        return

    region = get_capture_region(game_window)
    camera = dxcam.create(output_color="BGR")
    camera.start(region=region, target_fps=60, video_mode=True)
    setup_windows(game_window, args.minimap_w, args.minimap_h)

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    print("Live detection running. Press 'q' in any window to quit.")
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            gamma_corrected = cv2.LUT(frame, gamma_table)
            display_frame = cv2.convertScaleAbs(
                gamma_corrected, alpha=args.contrast, beta=args.brightness
            )

            h, w = display_frame.shape[:2]
            left_crop, top_crop, right_crop, bottom_crop = compute_minimap_crop(
                w,
                h,
                args.minimap_w,
                args.minimap_h,
                args.left_offset,
                args.top_offset,
                args.right_offset,
                args.bottom_offset,
            )
            minimap = display_frame[top_crop:bottom_crop, left_crop:right_crop].copy()
            results = model(minimap, conf=args.conf, verbose=False)
            annotated_minimap = draw_detections(minimap, results, CLASS_NAMES)
            display_frame[top_crop:bottom_crop, left_crop:right_crop] = annotated_minimap

            frame_count += 1
            if frame_count >= 30:
                now = time.time()
                fps = frame_count / (now - start_time)
                start_time = now
                frame_count = 0

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Full Game + Minimap Overlay", display_frame)
            cv2.imshow("Minimap Only", annotated_minimap)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
