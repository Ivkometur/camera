import time
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np


@dataclass
class Detection:
    conf: float
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2


class PeopleCounter:
    _CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    _PERSON_ID = _CLASSES.index("person")

    def __init__(
        self,
        prototxt_path: str,
        model_path: str,
        conf_thresh: float = 0.45,
        nms_thresh: float = 0.35,
        input_size: Tuple[int, int] = (300, 300),
    ):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.input_size = input_size

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_people(self, frame_bgr: np.ndarray) -> List[Detection]:
        (h, w) = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, self.input_size),
            scalefactor=0.007843,
            size=self.input_size,
            mean=127.5,
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        confs = []

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])
            if class_id != self._PERSON_ID:
                continue
            if conf < self.conf_thresh:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            boxes.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])  # x,y,w,h
            confs.append(conf)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)

        out: List[Detection] = []
        if len(idxs) > 0:
            for j in idxs.flatten():
                x, y, bw, bh = boxes[j]
                out.append(Detection(confs[j], (x, y, x + bw - 1, y + bh - 1)))
        return out

    def count_people(self, frame_bgr: np.ndarray) -> int:
        return len(self.detect_people(frame_bgr))
