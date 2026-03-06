import numpy as np
from pupil_apriltags import Detector

class AprilTagTracker:
    def __init__(self, families="tag36h11"):
        self.detector = Detector(
            families=families,
            nthreads=2,
            quad_decimate=1.0,   # increase to 2.0 for speed, but less accuracy
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.last_center = None
        self.last_seen_ts = None

    def detect(self, gray):
        # returns list of detections
        return self.detector.detect(gray)

    @staticmethod
    def best_detection(dets):
        # pick largest tag (by area proxy)
        if not dets:
            return None
        return max(dets, key=lambda d: float(d.decision_margin))

    @staticmethod
    def center_and_corners(det):
        c = det.center  # (x,y)
        corners = det.corners  # 4x2
        return (float(c[0]), float(c[1])), corners.astype(np.float32)