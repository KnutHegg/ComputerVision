import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class LogRow:
    t: float
    frame_idx: int
    w: int
    h: int
    decode_ms: float
    undistort_ms: float
    detect_ms: float
    total_ms: float
    fps_ema: float

    tag_id: Optional[int]
    decision_margin: Optional[float]
    cx: Optional[float]
    cy: Optional[float]
    steer: Optional[float]

    undistort_on: bool

    x_m: Optional[float]
    z_m: Optional[float]
    yaw_deg: Optional[float]


class JsonlLogger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "events.jsonl")
        self.f = open(self.path, "w", encoding="utf-8")

        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "jsonl",
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

    def write(self, row: LogRow):
        self.f.write(json.dumps(asdict(row)) + "\n")

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass