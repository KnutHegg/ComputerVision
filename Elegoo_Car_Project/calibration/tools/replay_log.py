import argparse
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_events(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help=r"Path to run folder, e.g. logs\run_20260301_013500")
    ap.add_argument("--show-video", action="store_true", help="Show recorded video with overlays")
    args = ap.parse_args()

    events_path = os.path.join(args.run, "events.jsonl")
    video_path = os.path.join(args.run, "video.mp4")

    rows = load_events(events_path)
    if not rows:
        raise RuntimeError("No events found")

    t0 = rows[0]["t"]
    t = np.array([r["t"] - t0 for r in rows], dtype=float)
    fps = np.array([r["fps_ema"] for r in rows], dtype=float)
    steer = np.array([np.nan if r["steer"] is None else r["steer"] for r in rows], dtype=float)
    margin = np.array([np.nan if r["decision_margin"] is None else r["decision_margin"] for r in rows], dtype=float)
    detected = np.array([0 if r["tag_id"] is None else 1 for r in rows], dtype=int)

    # plots
    plt.figure()
    plt.plot(t, fps)
    plt.xlabel("time (s)")
    plt.ylabel("fps (ema)")
    plt.title("FPS over time")
    plt.show()

    plt.figure()
    plt.plot(t, steer)
    plt.xlabel("time (s)")
    plt.ylabel("steer")
    plt.title("Steering suggestion over time")
    plt.show()

    plt.figure()
    plt.plot(t, margin)
    plt.xlabel("time (s)")
    plt.ylabel("decision margin")
    plt.title("Decision margin over time")
    plt.show()

    plt.figure()
    plt.step(t, detected, where="post")
    plt.xlabel("time (s)")
    plt.ylabel("detected (0/1)")
    plt.title("Detection over time")
    plt.ylim(-0.1, 1.1)
    plt.show()

    if args.show_video:
        if not os.path.exists(video_path):
            raise RuntimeError("No video.mp4 found (enable RECORD_VIDEO in run_tag_tracking.py)")

        cap = cv2.VideoCapture(video_path)
        i = 0
        print("Press q to quit video replay.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i < len(rows):
                r = rows[i]
                h, w = frame.shape[:2]
                cx_img = w // 2
                cv2.line(frame, (cx_img, 0), (cx_img, h), (255, 255, 255), 1)

                if r["cx"] is not None and r["cy"] is not None:
                    cv2.circle(frame, (int(r["cx"]), int(r["cy"])), 6, (0, 255, 0), -1)

                txt = f"id={r['tag_id']} margin={r['decision_margin']} steer={r['steer']} fps={r['fps_ema']:.1f}"
                cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("replay", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            i += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()