import yaml
import numpy as np
import cv2

class Undistorter:
    def __init__(self, calib_yaml: str):
        with open(calib_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        self.K = np.array(cfg["camera_matrix"], dtype=np.float64)
        self.dist = np.array(cfg["dist_coeffs"], dtype=np.float64)
        self.size = (int(cfg["image_width"]), int(cfg["image_height"]))  # (w,h)

        # alpha=0 => minimal black borders (cropped)
        self.newK, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, self.size, 0.0, self.size
        )

        # faster + stable remap
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.newK, self.size, cv2.CV_16SC2
        )

    def __call__(self, frame_bgr):
        und = cv2.remap(frame_bgr, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        if w > 0 and h > 0:
            und = und[y:y+h, x:x+w]
        return und