import argparse
import json
import os
import numpy as np


def load_events(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help=r"logs\run_...")
    ap.add_argument("--margin-thresh", type=float, default=20.0, help="Count detections with margin >= thresh")
    ap.add_argument("--lost-gap-s", type=float, default=0.35, help="Lost if no good detection for this long")
    args = ap.parse_args()

    events_path = os.path.join(args.run, "events.jsonl")
    rows = load_events(events_path)
    if not rows:
        raise RuntimeError("No events")

    t = np.array([r["t"] for r in rows], dtype=float)
    total_ms = np.array([r["total_ms"] for r in rows], dtype=float)
    fps = np.array([r["fps_ema"] for r in rows], dtype=float)

    margin = np.array([np.nan if r["decision_margin"] is None else r["decision_margin"] for r in rows], dtype=float)
    steer = np.array([np.nan if r["steer"] is None else r["steer"] for r in rows], dtype=float)

    good = np.isfinite(margin) & (margin >= args.margin_thresh)
    detection_rate = good.mean()

    # Latency stats
    p50_ms = float(np.percentile(total_ms, 50))
    p95_ms = float(np.percentile(total_ms, 95))

    # Jitter: steer std when detected
    if np.any(np.isfinite(steer) & good):
        steer_std = float(np.nanstd(steer[good]))
    else:
        steer_std = float("nan")

    # Lost / reacquisition: find gaps between good detections
    lost_events = 0
    reacq_times = []
    last_good_t = None
    in_lost = False
    lost_start = None

    for i in range(len(rows)):
        ti = t[i]
        if good[i]:
            if in_lost:
                reacq_times.append(ti - lost_start)
                in_lost = False
            last_good_t = ti
        else:
            if last_good_t is not None and (ti - last_good_t) >= args.lost_gap_s and not in_lost:
                in_lost = True
                lost_start = ti
                lost_events += 1

    avg_reacq = float(np.mean(reacq_times)) if reacq_times else float("nan")

    print("=== Evaluation ===")
    print(f"Frames: {len(rows)}")
    print(f"Detection rate (margin>={args.margin_thresh}): {detection_rate*100:.1f}%")
    print(f"Latency total_ms: p50={p50_ms:.1f} ms, p95={p95_ms:.1f} ms")
    print(f"FPS (EMA): mean={float(np.mean(fps)):.1f}, min={float(np.min(fps)):.1f}, max={float(np.max(fps)):.1f}")
    print(f"Steering jitter (std, when detected): {steer_std:.3f}")
    print(f"Lost events (gap>={args.lost_gap_s}s): {lost_events}")
    print(f"Avg reacquisition time: {avg_reacq:.3f} s")


if __name__ == "__main__":
    main()