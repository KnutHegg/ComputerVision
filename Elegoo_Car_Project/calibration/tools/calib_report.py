import argparse
import glob
import os
import random
from datetime import datetime

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def undistort_image(img_bgr, K, dist):
    h, w = img_bgr.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0, (w, h))
    und = cv2.undistort(img_bgr, K, dist, None, newK)
    return und


def save_side_by_side(before_bgr, after_bgr, out_path):
    # same height already; stack horizontally
    vis = np.hstack([before_bgr, after_bgr])
    cv2.imwrite(out_path, vis, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=r"configs\camera.yaml", help="Calibration yaml path")
    ap.add_argument("--frames", required=True, help=r"Frames folder, e.g. data\calib\session_x\frames")
    ap.add_argument("--out", default=r"reports\calibration_report.html", help="Output html report")
    ap.add_argument("--samples", type=int, default=6, help="How many preview images to include")
    args = ap.parse_args()

    cfg = load_yaml(args.calib)
    K = np.array(cfg["camera_matrix"], dtype=np.float64)
    dist = np.array(cfg["dist_coeffs"], dtype=np.float64)
    errs = np.array(cfg["per_view_errors_px"], dtype=float)

    rms = float(cfg.get("rms_reprojection_error", float("nan")))
    used_count = int(cfg.get("used_images_count", -1))

    mean_err = float(errs.mean()) if len(errs) else float("nan")
    p95_err = float(np.quantile(errs, 0.95)) if len(errs) else float("nan")

    # Create report folder + assets
    out_dir = os.path.dirname(args.out)
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    assets_dir = os.path.join(out_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Error histogram plot
    hist_path = os.path.join(assets_dir, "reproj_hist.png")
    fig = plt.figure()
    plt.hist(errs, bins=20)
    plt.title("Per-view reprojection error (px)")
    plt.xlabel("px")
    plt.ylabel("count")
    fig.savefig(hist_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Undistortion samples
    imgs = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) + glob.glob(os.path.join(args.frames, "*.png")))
    if not imgs:
        raise RuntimeError(f"No images found in: {args.frames}")

    random.shuffle(imgs)
    sample_imgs = imgs[: args.samples]

    preview_relpaths = []
    for i, p in enumerate(sample_imgs):
        img = cv2.imread(p)
        if img is None:
            continue
        und = undistort_image(img, K, dist)
        out_png = os.path.join(assets_dir, f"undistort_{i}.png")
        save_side_by_side(img, und, out_png)
        preview_relpaths.append(os.path.relpath(out_png, out_dir))

    status = "PASS" if (rms < 2.0 and p95_err < 1.0) else "REVIEW"

    params_subset = {
        "image_width": cfg.get("image_width"),
        "image_height": cfg.get("image_height"),
        "camera_matrix": cfg.get("camera_matrix"),
        "dist_coeffs": cfg.get("dist_coeffs"),
    }

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Camera Calibration Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .kpi {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0 24px; }}
    .card {{ padding: 12px 16px; border: 1px solid #ddd; border-radius: 10px; }}
    .status {{ font-weight: bold; }}
    img {{ max-width: 100%; border: 1px solid #eee; border-radius: 8px; margin: 8px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    pre {{ background: #fafafa; padding: 12px; border-radius: 8px; border: 1px solid #eee; overflow-x: auto; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Camera Calibration Report</h1>
  <p class="muted">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <div class="kpi">
    <div class="card"><div class="status">Status: {status}</div></div>
    <div class="card">Used images: <b>{used_count}</b></div>
    <div class="card">RMS reprojection error: <b>{rms:.3f}px</b></div>
    <div class="card">Mean per-view error: <b>{mean_err:.3f}px</b></div>
    <div class="card">P95 per-view error: <b>{p95_err:.3f}px</b></div>
  </div>

  <h2>Parameters</h2>
  <pre>{yaml.safe_dump(params_subset, sort_keys=False)}</pre>

  <h2>Error distribution</h2>
  <img src="{os.path.relpath(hist_path, out_dir)}" alt="reprojection histogram"/>

  <h2>Undistortion samples</h2>
  <p class="muted">Left = raw, right = undistorted</p>
  <div class="grid">
    {''.join([f'<img src="{p}" alt="undistort sample"/>' for p in preview_relpaths])}
  </div>

  <p class="muted" style="margin-top:24px;">
    Notes: If status is REVIEW, try recording more frames with the board near image edges, different tilts, and less motion blur.
  </p>
</body>
</html>
"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print("[report] Wrote:", args.out)


if __name__ == "__main__":
    main()