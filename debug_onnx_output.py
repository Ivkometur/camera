import os
import json
import numpy as np
import cv2

from dotenv import load_dotenv

from rtsp_snapshot import rtsp_snapshot_bgr
from camera_people_yolo import apply_office_mask_2x3, letterbox, _sigmoid


def load_env():
    # грузим /opt/factory-bot/.env всегда
    load_dotenv("/opt/factory-bot/.env", override=False)


def pick_rtsp_url() -> str:
    # 1) прямой RTSP_URL
    rtsp = (os.getenv("RTSP_URL") or "").strip()
    if rtsp:
        return rtsp

    # 2) из CAMERAS_JSON
    cams_json = (os.getenv("CAMERAS_JSON") or "").strip()
    if cams_json:
        cams = json.loads(cams_json)
        if isinstance(cams, list) and cams:
            for c in cams:
                if str(c.get("id", "")).strip().lower() == "office" and c.get("rtsp_url"):
                    return str(c["rtsp_url"]).strip()
            # fallback: first
            if "rtsp_url" in cams[0]:
                return str(cams[0]["rtsp_url"]).strip()

    raise SystemExit("No RTSP URL found. Set RTSP_URL or CAMERAS_JSON in /opt/factory-bot/.env")


def main():
    load_env()

    model_path = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.onnx")
    imgsz = int(os.getenv("YOLO_IMGSZ", "640"))

    rtsp = pick_rtsp_url()
    print("RTSP:", rtsp)

    frame = rtsp_snapshot_bgr(rtsp)
    H, W = frame.shape[:2]
    print("FRAME:", W, H)

    masked = apply_office_mask_2x3(frame)
    img, r, (dw, dh) = letterbox(masked, imgsz)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)

    net = cv2.dnn.readNetFromONNX(model_path)
    net.setInput(blob)
    outs = net.forward()

    if isinstance(outs, (list, tuple)):
        print("forward() returned list:", [o.shape for o in outs])
        out = outs[0]
    else:
        print("forward() returned:", outs.shape)
        out = outs

    os.makedirs("/opt/factory-bot/photos", exist_ok=True)
    cv2.imwrite("/opt/factory-bot/photos/dbg_orig.jpg", frame)
    cv2.imwrite("/opt/factory-bot/photos/dbg_masked.jpg", masked)

    # normalize to 2D
    out2 = out[0] if out.ndim == 3 else out
    print("OUT ndim:", out.ndim, "-> out2 shape:", out2.shape, "dtype:", out2.dtype)

    arr = out2.astype(np.float32).ravel()
    print("OUT min/max/mean:", float(arr.min()), float(arr.max()), float(arr.mean()))

    cand = out2
    if cand.shape[0] in (84, 85) and cand.shape[1] > 1000:
        cand = cand.T
        print("Heuristic transpose ->", cand.shape)

    if cand.ndim == 2 and cand.shape[1] >= 84:
        xywh = cand[:, 0:4]
        cls = cand[:, 4:84]

        mx = float(np.max(cls))
        print("CLS max before sigmoid:", mx)
        if mx > 1.5:
            cls = _sigmoid(cls)
            print("Applied sigmoid to class scores")

        person = cls[:, 0]
        top_idx = np.argsort(-person)[:5]
        print("Top-5 by person score idx:", top_idx.tolist())
        for i in top_idx:
            i = int(i)
            p = float(person[i])
            best_c = int(np.argmax(cls[i]))
            best_s = float(cls[i, best_c])
            print(f"  i={i} person={p:.4f} best_class={best_c} best_score={best_s:.4f} xywh={xywh[i].tolist()}")
    else:
        print("Cannot interpret output as YOLOv8 (N,84). Shape:", cand.shape)


if __name__ == "__main__":
    main()
