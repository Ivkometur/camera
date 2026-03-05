import os, time
from rtsp_snapshot import rtsp_snapshot_bgr
from camera_people_ultra import PeopleCounterUltra

cams_json = os.getenv("CAMERAS_JSON","[]")
# monitor_office.py сам выбирает камеру, но тут проще: возьмём OFFICE_RTSP_URL если есть
rtsp = (os.getenv("OFFICE_RTSP_URL","") or "").strip()
if not rtsp:
    raise SystemExit("Set OFFICE_RTSP_URL in env for debug_once_ultra.py (temporary).")

counter = PeopleCounterUltra()
frame = rtsp_snapshot_bgr(rtsp)
res = counter.infer(frame)

ts = int(time.time())
prefix = f"/opt/factory-bot/photos/once_{ts}"
paths = counter.draw_debug(frame, res.get("masked"), res.get("detections") or [], prefix)

print("count=", res.get("count"))
print("stats=", res.get("stats"))
print("saved:", paths)
