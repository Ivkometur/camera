import os, json, time, statistics
from dotenv import load_dotenv

from rtsp_snapshot import rtsp_snapshot_bgr
from camera_people_yolo import PeopleCounterYOLO

def pick_rtsp():
    cams = json.loads(os.getenv("CAMERAS_JSON","[]"))
    for c in cams:
        if str(c.get("id","")).lower()=="office":
            return c["rtsp_url"]
    if cams:
        return cams[0]["rtsp_url"]
    raise SystemExit("CAMERAS_JSON is empty")

def main():
    load_dotenv("/opt/factory-bot/.env")
    rtsp = pick_rtsp()
    model = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.onnx")
    pc = PeopleCounterYOLO(model)

    counts = []
    last = None
    for i in range(5):
        frame = rtsp_snapshot_bgr(rtsp)
        res = pc.infer(frame)
        counts.append(int(res["count"]))
        last = (frame, res)
        time.sleep(0.12)

    med = int(statistics.median(counts))
    print("counts:", counts, "median:", med)
    if last:
        frame, res = last
        print("last_count:", res["count"])
        print("stats:", res["stats"])

        os.makedirs("/opt/factory-bot/photos", exist_ok=True)
        ts = int(time.time())
        out_prefix = f"/opt/factory-bot/photos/test_{ts}"
        paths = pc.draw_debug(frame, res["masked"], res["detections"], out_prefix)
        print("saved:", paths)

if __name__ == "__main__":
    main()
