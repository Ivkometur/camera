import os, json, cv2
from dotenv import load_dotenv
from rtsp_snapshot import grab_frame_ffmpeg

load_dotenv("/opt/factory-bot/.env")

prototxt = "models/MobileNetSSD_deploy.prototxt"
model = "models/MobileNetSSD_deploy.caffemodel"
conf_thresh = float(os.getenv("DETECT_CONF","0.45"))

cams = json.loads(os.getenv("CAMERAS_JSON"))
cid = os.getenv("OFFICE_CAMERA_ID","office")
cam = [c for c in cams if c.get("id")==cid][0]
img = grab_frame_ffmpeg(cam["rtsp_url"])

(h, w) = img.shape[:2]
net = cv2.dnn.readNetFromCaffe(prototxt, model)

blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

person_class_id = 15  # for this MobileNet-SSD
count = 0
rows = []

for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])
    cls = int(detections[0, 0, i, 1])
    if cls != person_class_id:
        continue
    if confidence < conf_thresh:
        continue
    box = detections[0, 0, i, 3:7] * [w, h, w, h]
    (x1, y1, x2, y2) = box.astype("int")
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"person {confidence:.2f}", (x1, max(15, y1-7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    count += 1
    rows.append((confidence, x1, y1, x2, y2))

out = "photos/debug_boxes.jpg"
cv2.imwrite(out, img)

print("detected_people =", count, "conf_thresh =", conf_thresh)
for conf, x1, y1, x2, y2 in sorted(rows, reverse=True)[:10]:
    print(f"  conf={conf:.2f} box=({x1},{y1})-({x2},{y2})")
print("saved:", out)
