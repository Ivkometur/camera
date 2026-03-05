import os
import time
import statistics
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


class PeopleCounterUltra:
    """
    Ultralytics YOLOv8 counter.
    - counts class=0 (person)
    - supports debug draw
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = (model_name or os.getenv("ULTRA_MODEL", "yolov8m.pt")).strip() or "yolov8m.pt"
        self.conf = _env_float("ULTRA_CONF", 0.25)
        self.imgsz = _env_int("ULTRA_IMGSZ", 960)

        self.model = YOLO(self.model_name)

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = self.model.predict(rgb, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]

        dets: List[Dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                if cls[i] == 0:  # person
                    x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                    dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(sc[i])})

        stats = {
            "engine": "ultra",
            "frame_wh": (int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "model": self.model_name,
            "people_raw": len(dets),
        }
        return {"count": len(dets), "detections": dets, "stats": stats, "masked": frame_bgr}

    def draw_debug(self, frame_bgr: np.ndarray, masked_bgr: np.ndarray, dets: List[Dict[str, Any]], out_prefix: str) -> Dict[str, str]:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        orig_path = f"{out_prefix}_orig.jpg"
        masked_path = f"{out_prefix}_masked.jpg"
        boxes_path = f"{out_prefix}_boxes.jpg"

        cv2.imwrite(orig_path, frame_bgr)
        cv2.imwrite(masked_path, masked_bgr)

        img = masked_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
            sc = float(d.get("score", 0.0))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{sc:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(boxes_path, img)
        return {"orig": orig_path, "masked": masked_path, "boxes": boxes_path}
