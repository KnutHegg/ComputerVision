import argparse
import glob
import os

import cv2
import numpy as np
import yaml


def find_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    # subpixel refine
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help=r"Path to frames folder, e.g. data\calib\session_x\frames")
    ap.add_argument("--pattern", required=True, help="Inner corners WxH, e.g. 7x7")
    ap.add_argument("--square-mm", type=float, required=True, help="Square size in mm, e.g. 35")
    ap.add_argument("--out", default=r"configs\camera.yaml", help="Output yaml path")
    args = ap.parse_args()

    w, h = args.pattern.lower().split("x")
    pattern = (int(w), int(h))
    square_size = args.square_mm / 1000.0  # meters

    images = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) + glob.glob(os.path.join(args.frames, "*.png")))
    if not images:
        raise RuntimeError(f"No images found in: {args.frames}")

    # object points grid
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints, used = [], [], []
    image_size = None

    for p in images:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            H, W = gray.shape[:2]
            image_size = (W, H)

        corners = find_corners(gray, pattern)
        if corners is None:
            continue

        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(p)

    if len(used) < 12:
        raise RuntimeError(
            f"Only detected the chessboard in {len(used)} images.\n"
            "Record more frames with clearer corners / varied angles, or check --pattern is correct."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # per-view reprojection errors
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_view.append(float(err))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rms_reprojection_error": float(rms),
        "per_view_errors_px": per_view,
        "used_images_count": int(len(used)),
    }

    with open(args.out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("[calib] Saved:", args.out)
    print("[calib] Used images:", len(used), "/", len(images))
    print("[calib] RMS reprojection error:", float(rms))
    print("[calib] Per-view error px: mean =", float(np.mean(per_view)), "p95 =", float(np.quantile(per_view, 0.95)))


if __name__ == "__main__":
    main()