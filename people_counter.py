import os
import time
from typing import Dict, Any, Tuple, List

import numpy as np

from rtsp_snapshot import rtsp_snapshot_bgr

# OpenCV-ONNX counter (твой текущий)
from camera_people_yolo import PeopleCounterYOLO

# Ultralytics counter
from people_counter_ultra import PeopleCounterUltra


def _engine() -> str:
    return (os.getenv("DETECTOR_ENGINE") or "opencv").strip().lower()


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


class PeopleCounter:
    def __init__(self):
        self.engine = _engine()
        if self.engine == "ultra":
            self.impl = PeopleCounterUltra()
        else:
            model_path = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.onnx")
            self.impl = PeopleCounterYOLO(model_path)

    def snapshot_frames(self, rtsp_url: str, n: int = 5, sleep_s: float = 0.12) -> List[np.ndarray]:
        frames = []
        for i in range(n):
            frames.append(rtsp_snapshot_bgr(rtsp_url))
            time.sleep(max(0.0, sleep_s))
        return frames

    def count(self, rtsp_url: str) -> Tuple[int, Dict[str, Any]]:
        n = _env_int("ULTRA_N", 5) if self.engine == "ultra" else _env_int("YOLO_N", 1)
        frames = self.snapshot_frames(rtsp_url, n=n, sleep_s=0.12)

        if self.engine == "ultra":
            med, st, _ = self.impl.count_people_median(frames)
            return med, st
        else:
            # opencv engine counts last frame only (у тебя уже есть тайлы+median в другом месте; пока так)
            res = self.impl.infer(frames[-1])
            st = res.get("stats") or {}
            st["engine"] = "opencv"
            return int(res.get("count", 0)), st
