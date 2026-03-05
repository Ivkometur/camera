import os, json, time, statistics
from dotenv import load_dotenv
import cv2

from ultralytics import YOLO
from rtsp_snapshot import rtsp_snapshot_bgr

def pick_rtsp():
    cams = json.loads(os.getenv("CAMERAS_JSON","[]"))
    for c in cams:
        if str(c.get("id","")).lower()=="office":
            return c["rtsp_url"]
    return cams[0]["rtsp_url"]

def main():
    load_dotenv("/opt/factory-bot/.env")
    rtsp = pick_rtsp()

    conf = float(os.getenv("ULTRA_CONF","0.25"))
    imgsz = int(os.getenv("ULTRA_IMGSZ","960"))  # тут уже можно 960/1280!
    n = int(os.getenv("ULTRA_N","5"))

    model = YOLO("yolov8m.pt")

    counts = []
    last_frame = None
    last_boxes = None

    for i in range(n):
        frame = rtsp_snapshot_bgr(rtsp)
        last_frame = frame

        # Ultralytics expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = model.predict(rgb, imgsz=imgsz, conf=conf, verbose=False)[0]

        # class 0 = person
        if r.boxes is None or len(r.boxes) == 0:
            counts.append(0)
            last_boxes = []
        else:
            boxes = r.boxes
            cls = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            people = [(xyxy[i], float(scores[i])) for i in range(len(cls)) if cls[i] == 0]
            counts.append(len(people))
            last_boxes = people

        time.sleep(0.12)

    med = int(statistics.median(counts)) if counts else 0
    print("counts:", counts, "median:", med, "conf:", conf, "imgsz:", imgsz)

    # save debug
    os.makedirs("/opt/factory-bot/photos", exist_ok=True)
    ts = int(time.time())
    out = last_frame.copy()
    for (xy, sc) in (last_boxes or []):
        x1,y1,x2,y2 = [int(v) for v in xy]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"{sc:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    path = f"/opt/factory-bot/photos/ultra_{ts}.jpg"
    cv2.imwrite(path, out)
    print("saved:", path)

if __name__ == "__main__":
    main()
