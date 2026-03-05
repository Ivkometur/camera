import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def apply_office_mask_2x3(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    row_h = h // 2
    col_w = w // 3
    # верхние 2 левые зоны
    out[0:row_h, 0:col_w] = 0
    out[0:row_h, col_w:2 * col_w] = 0
    return out


def _xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    order = scores.argsort()[::-1].tolist()
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            if _xyxy_iou(boxes[i], boxes[j]) <= iou_th:
                rest.append(j)
        order = rest
    return keep


class PeopleCounterUltra:
    def __init__(self):
        self.model_name = (os.getenv("ULTRA_MODEL", "yolov8n.pt") or "").strip() or "yolov8n.pt"
        self.imgsz = _env_int("ULTRA_IMGSZ", _env_int("YOLO_IMGSZ", 960))
        self.conf = _env_float("ULTRA_CONF", _env_float("YOLO_CONF", 0.25))
        self.iou = _env_float("ULTRA_IOU", _env_float("YOLO_IOU", 0.45))

        self.min_h_frac = _env_float("FILTER_MIN_H_FRAC", 0.05)
        self.min_area_frac = _env_float("FILTER_MIN_AREA_FRAC", 0.001)
        self.min_aspect = _env_float("FILTER_ASPECT", 0.85)

        self.no_filters = _env_bool("DEBUG_NO_FILTERS", False)
        self.apply_mask = _env_bool("APPLY_OFFICE_MASK", True)

        self.model = YOLO(self.model_name)

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        H, W = frame_bgr.shape[:2]

        if self.apply_mask:
            masked = apply_office_mask_2x3(frame_bgr)
        else:
            masked = frame_bgr

        rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

        res_list = self.model.predict(
            source=rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[0],  # person
            verbose=False,
            device="cpu",
        )
        r = res_list[0]

        dets: List[Dict[str, Any]] = []
        raw_n = 0

        min_h_px = int(self.min_h_frac * H)
        min_area_px = int(self.min_area_frac * (W * H))
        dropped = {"min_h": 0, "min_area": 0, "aspect": 0}

        if r.boxes is not None and len(r.boxes) > 0:
            raw_n = len(r.boxes)
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32)

            keep = _nms_xyxy(xyxy, confs, self.iou)
            xyxy = xyxy[keep]
            confs = confs[keep]

            for bb, sc in zip(xyxy, confs):
                x1, y1, x2, y2 = [float(v) for v in bb.tolist()]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h
                aspect = (h / w) if w > 1e-6 else 999.0

                if not self.no_filters:
                    if h < min_h_px:
                        dropped["min_h"] += 1
                        continue
                    if area < min_area_px:
                        dropped["min_area"] += 1
                        continue
                    if aspect < self.min_aspect:
                        dropped["aspect"] += 1
                        continue

                dets.append({"xyxy": [x1, y1, x2, y2], "score": float(sc)})

        stats = {
            "frame_wh": (W, H),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "raw_after_conf_sum": int(raw_n),
            "kept_after_filters": int(len(dets)),
            "apply_mask": bool(self.apply_mask),
            "filters": {
                "min_h_frac": self.min_h_frac,
                "min_area_frac": self.min_area_frac,
                "min_aspect": self.min_aspect,
                "no_filters": bool(self.no_filters),
            },
            "dropped": dropped,
            "min_h_px": min_h_px,
            "min_area_px": min_area_px,
            "t_ms": int((time.time() - t0) * 1000),
        }

        return {"count": int(len(dets)), "detections": dets, "stats": stats, "masked": masked}

    def draw_debug(self, orig_bgr: np.ndarray, masked_bgr: np.ndarray, dets: List[Dict[str, Any]], out_prefix: str) -> Dict[str, str]:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        p_orig = f"{out_prefix}_orig.jpg"
        p_mask = f"{out_prefix}_masked.jpg"
        p_boxes = f"{out_prefix}_boxes.jpg"

        cv2.imwrite(p_orig, orig_bgr)
        cv2.imwrite(p_mask, masked_bgr)

        vis = orig_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            sc = float(d.get("score", 0.0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"person {sc:.2f}", (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(p_boxes, vis)
        return {"orig": p_orig, "masked": p_mask, "boxes": p_boxes}
