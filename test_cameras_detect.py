import os, json
from dotenv import load_dotenv

from camera_people import PeopleCounter
from rtsp_snapshot import grab_frame_ffmpeg

load_dotenv()

raw = (os.getenv("CAMERAS_JSON") or "").strip()
if not raw:
    raise SystemExit("CAMERAS_JSON is not set in .env")

try:
    cameras = json.loads(raw)
except Exception as e:
    raise SystemExit(f"CAMERAS_JSON is not valid JSON: {e}")

conf = float(os.getenv("DETECT_CONF", "0.45"))
nms = float(os.getenv("DETECT_NMS", "0.35"))

counter = PeopleCounter(
    prototxt_path="models/MobileNetSSD_deploy.prototxt",
    model_path="models/MobileNetSSD_deploy.caffemodel",
    conf_thresh=conf,
    nms_thresh=nms,
)

for cam in cameras:
    if not cam.get("enabled", True):
        continue
    cam_id = cam.get("id", "?")
    name = cam.get("name", cam_id)
    room = cam.get("room", "")
    rtsp = cam.get("rtsp_url", "")
    if not rtsp:
        print(f"[{cam_id}] {name}: rtsp_url missing, skip")
        continue

    try:
        frame = grab_frame_ffmpeg(rtsp)
        n = counter.count_people(frame)
        print(f"[{cam_id}] {name} ({room}): detected_people = {n}")
    except Exception as e:
        print(f"[{cam_id}] {name} ({room}): ERROR: {e}")
