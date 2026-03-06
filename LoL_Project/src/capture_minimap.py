import argparse
import time
from pathlib import Path

import cv2
import dxcam
import pygetwindow as gw

from common import PROJECT_ROOT, compute_minimap_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture raw minimap crops from live LoL window.")
    parser.add_argument("--window-title", type=str, default="League of Legends (TM) Client")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "raw_data" / "real_minimaps")
    parser.add_argument("--capture-interval", type=float, default=10.0)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--minimap-w", type=int, default=300)
    parser.add_argument("--minimap-h", type=int, default=300)
    parser.add_argument("--left-offset", type=int, default=32)
    parser.add_argument("--top-offset", type=int, default=32)
    parser.add_argument("--right-offset", type=int, default=16)
    parser.add_argument("--bottom-offset", type=int, default=11)
    return parser.parse_args()


def get_game_window(window_title: str):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise RuntimeError(f"Game window not found: {window_title}")
    return windows[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        game_window = get_game_window(args.window_title)
    except RuntimeError as error:
        print(error)
        return

    left = game_window.left
    top = game_window.top
    right = left + game_window.width
    bottom = top + game_window.height
    region = (left, top, right, bottom)

    camera = dxcam.create(output_color="BGR")
    camera.start(region=region, target_fps=30, video_mode=True)
    print(f"Saving minimap captures to: {args.output_dir}")
    print("Press 'q' in the preview window to stop.")

    count = 0
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
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
            minimap = frame[top_crop:bottom_crop, left_crop:right_crop]

            out_path = args.output_dir / f"minimap_{count:06d}.jpg"
            cv2.imwrite(str(out_path), minimap)
            print(f"Saved: {out_path}")
            count += 1

            cv2.imshow("Minimap Capture", minimap)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.max_images and count >= args.max_images:
                break
            time.sleep(args.capture_interval)
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

