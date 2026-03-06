import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests


def mjpeg_frames(url: str, timeout=10):
    """
    Generator that yields decoded BGR frames from an MJPEG HTTP stream.
    Works well with ESP32-CAM style endpoints like http://<ip>:81/stream
    """
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    buffer = b""
    for chunk in r.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buffer += chunk

        # Look for JPEG start/end markers
        a = buffer.find(b"\xff\xd8")  # SOI
        b = buffer.find(b"\xff\xd9")  # EOI
        if a != -1 and b != -1 and b > a:
            jpg = buffer[a : b + 2]
            buffer = buffer[b + 2 :]

            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://192.168.4.1:81/stream")
    ap.add_argument("--out", default=None, help="Output folder (default: data/calib/session_<timestamp>)")
    ap.add_argument("--every", type=int, default=10, help="Save every Nth decoded frame")
    ap.add_argument("--max", type=int, default=200, help="Max images to save")
    ap.add_argument("--preview", action="store_true", help="Show preview window")
    args = ap.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join("data", "calib", f"session_{ts}")

    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"[record] Stream: {args.url}")
    print(f"[record] Saving to: {frames_dir}")
    print("[record] Press 'q' in the preview window to stop early (if preview is on).")

    saved = 0
    decoded = 0
    last_save_time = time.time()

    try:
        for frame in mjpeg_frames(args.url):
            decoded += 1

            if args.preview:
                vis = frame.copy()
                cv2.putText(vis, f"decoded={decoded} saved={saved}/{args.max}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("record_stream", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if decoded % args.every == 0:
                path = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
                ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    saved += 1
                    last_save_time = time.time()
                    print(f"[record] saved {saved}/{args.max}: {path}")

            if saved >= args.max:
                break

            # If nothing gets saved for a while, hint at connectivity issues
            if time.time() - last_save_time > 15:
                print("[record] Still decoding but not saving—check --every value or stream stability.")
                last_save_time = time.time()

    except requests.RequestException as e:
        print(f"[record] Stream error: {e}")
    finally:
        cv2.destroyAllWindows()

    print(f"[record] Done. Decoded frames: {decoded}, saved images: {saved}")


if __name__ == "__main__":
    main()