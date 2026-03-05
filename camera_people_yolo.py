import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def letterbox(im: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw //= 2
    dh //= 2

    im_padded = cv2.copyMakeBorder(
        im_resized,
        dh, new_shape - new_unpad[1] - dh,
        dw, new_shape - new_unpad[0] - dw,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return im_padded, r, (dw, dh)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.copy(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def apply_office_mask_2x3(frame_bgr: np.ndarray) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    cell_w = w // 3
    cell_h = h // 2

    # закрываем верхний ряд: col 0 и col 1
    x1, y1 = 0, 0
    x2, y2 = cell_w * 2, cell_h
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    return img


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class PeopleCounterYOLO:
    """
    YOLOv8n ONNX (OpenCV DNN).
    ВАЖНО: твой ONNX фиксирован под 640x640 — YOLO_IMGSZ менять нельзя.
    Улучшение качества на 2K делаем тайлингом (TILE_INFER=1).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # model params
        self.imgsz = _env_int("YOLO_IMGSZ", 640)
        self.conf = _env_float("YOLO_CONF", 0.45)
        self.iou = _env_float("YOLO_IOU", 0.45)

        # debug
        self.no_filters = _env_bool("DEBUG_NO_FILTERS", False)

        # filters
        self.min_h_frac = _env_float("FILTER_MIN_H_FRAC", 0.05)
        self.min_area_frac = _env_float("FILTER_MIN_AREA_FRAC", 0.0010)
        self.min_aspect = _env_float("FILTER_ASPECT", 0.85)

        # tiling
        self.tile_infer = _env_bool("TILE_INFER", True)
        self.tile_cols = _env_int("TILE_COLS", 2)
        self.tile_rows = _env_int("TILE_ROWS", 2)
        self.tile_overlap = _env_float("TILE_OVERLAP", 0.18)

    def _decode(self, out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] in (84, 85) and out.shape[1] > 1000:
            out = out.T  # (N,84)

        xywh = out[:, 0:4].astype(np.float32)
        cls_scores = out[:, 4:].astype(np.float32)

        mx = float(np.max(cls_scores))
        if mx > 1.5:
            cls_scores = _sigmoid(cls_scores)

        scores = cls_scores[:, 0]  # person class 0
        return xywh, scores

    def _infer_single(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        h, w = bgr.shape[:2]
        img, r, (dw, dh) = letterbox(bgr, self.imgsz)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward()
        out = outs[0] if isinstance(outs, (list, tuple)) else outs

        xywh, scores = self._decode(out)

        keep = scores >= self.conf
        xywh = xywh[keep]
        scores = scores[keep]

        stats = {"raw_after_conf": int(xywh.shape[0])}

        if xywh.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), stats

        boxes = xywh_to_xyxy(xywh)
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
        boxes = clip_boxes(boxes, w, h)

        b_list = boxes.tolist()
        s_list = scores.tolist()
        idxs = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in b_list],
            scores=s_list,
            score_threshold=float(self.conf),
            nms_threshold=float(self.iou),
        )
        if len(idxs) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), stats

        idxs = idxs.flatten().tolist() if isinstance(idxs, np.ndarray) else [int(i) for i in idxs]
        boxes = boxes[idxs]
        scores = scores[idxs]
        stats["kept_after_nms"] = int(boxes.shape[0])
        return boxes.astype(np.float32), scores.astype(np.float32), stats

    def _make_tiles(self, W: int, H: int) -> List[Tuple[int, int, int, int]]:
        cols = max(1, int(self.tile_cols))
        rows = max(1, int(self.tile_rows))
        ov = float(self.tile_overlap)

        tile_w = int(np.ceil(W / cols))
        tile_h = int(np.ceil(H / rows))

        step_w = max(1, int(tile_w * (1.0 - ov)))
        step_h = max(1, int(tile_h * (1.0 - ov)))

        tiles = []
        y = 0
        while y < H:
            x = 0
            y2 = min(H, y + tile_h)
            while x < W:
                x2 = min(W, x + tile_w)
                tiles.append((x, y, x2, y2))
                if x2 >= W:
                    break
                x += step_w
            if y2 >= H:
                break
            y += step_h
        return tiles

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        H, W = frame_bgr.shape[:2]
        masked = apply_office_mask_2x3(frame_bgr)

        all_boxes = []
        all_scores = []
        raw_after_conf_sum = 0

        if self.tile_infer and (W > self.imgsz or H > self.imgsz):
            tiles = self._make_tiles(W, H)
            for (x1, y1, x2, y2) in tiles:
                tile = masked[y1:y2, x1:x2]
                boxes, scores, st = self._infer_single(tile)
                raw_after_conf_sum += int(st.get("raw_after_conf", 0))
                if boxes.shape[0] == 0:
                    continue
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_boxes.append(boxes)
                all_scores.append(scores)

            if all_boxes:
                boxes = np.concatenate(all_boxes, axis=0)
                scores = np.concatenate(all_scores, axis=0)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes, scores, st = self._infer_single(masked)
            raw_after_conf_sum = int(st.get("raw_after_conf", 0))

        stats = {
            "frame_wh": (W, H),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "tile_infer": self.tile_infer,
            "tile_grid": (self.tile_cols, self.tile_rows),
            "tile_overlap": self.tile_overlap,
            "raw_after_conf_sum": raw_after_conf_sum,
            "no_filters": self.no_filters,
            "filters": {"min_h_frac": self.min_h_frac, "min_area_frac": self.min_area_frac, "min_aspect": self.min_aspect},
            "dropped": {"min_h": 0, "min_area": 0, "aspect": 0},
        }

        if boxes.shape[0] == 0:
            stats["t_ms"] = int((time.time() - t0) * 1000)
            return {"count": 0, "detections": [], "stats": stats, "masked": masked}

        # global NMS
        b_list = boxes.tolist()
        s_list = scores.tolist()
        idxs = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in b_list],
            scores=s_list,
            score_threshold=float(self.conf),
            nms_threshold=float(self.iou),
        )
        if len(idxs) == 0:
            stats["t_ms"] = int((time.time() - t0) * 1000)
            return {"count": 0, "detections": [], "stats": stats, "masked": masked}

        idxs = idxs.flatten().tolist() if isinstance(idxs, np.ndarray) else [int(i) for i in idxs]
        boxes = boxes[idxs]
        scores = scores[idxs]
        stats["kept_after_global_nms"] = int(boxes.shape[0])

        # filters
        min_h = int(H * self.min_h_frac)
        min_area = int(W * H * self.min_area_frac)

        dets: List[Detection] = []
        for b, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            area = w_box * h_box
            aspect = h_box / float(w_box)

            if not self.no_filters:
                if h_box < min_h:
                    stats["dropped"]["min_h"] += 1
                    continue
                if area < min_area:
                    stats["dropped"]["min_area"] += 1
                    continue
                if aspect < self.min_aspect:
                    stats["dropped"]["aspect"] += 1
                    continue

            dets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=float(sc)))

        stats["kept_after_filters"] = len(dets)
        stats["min_h_px"] = min_h
        stats["min_area_px"] = min_area
        stats["t_ms"] = int((time.time() - t0) * 1000)

        return {"count": len(dets), "detections": dets, "stats": stats, "masked": masked}

    def draw_debug(self, frame_bgr: np.ndarray, masked_bgr: np.ndarray, dets: List[Detection], out_prefix: str) -> Dict[str, str]:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        orig_path = f"{out_prefix}_orig.jpg"
        masked_path = f"{out_prefix}_masked.jpg"
        boxes_path = f"{out_prefix}_boxes.jpg"

        cv2.imwrite(orig_path, frame_bgr)
        cv2.imwrite(masked_path, masked_bgr)

        img = masked_bgr.copy()
        for d in dets:
            cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d.score:.2f}", (d.x1, max(0, d.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(boxes_path, img)

        return {"orig": orig_path, "masked": masked_path, "boxes": boxes_path}
