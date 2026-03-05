import os, json, time
import cv2
import numpy as np
from dotenv import load_dotenv
from rtsp_snapshot import grab_frame_ffmpeg
from camera_people_yolo import apply_office_mask_2x3, _sigmoid  # берём ту же маску/сигмоиду

def letterbox(img, new_size=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = new_size - nw
    pad_h = new_size - nh
    left = pad_w // 2
    top = pad_h // 2
    out = cv2.copyMakeBorder(resized, top, pad_h-top, left, pad_w-left, cv2.BORDER_CONSTANT, value=color)
    return out, r, left, top

load_dotenv("/opt/factory-bot/.env")
cams = json.loads(os.getenv("CAMERAS_JSON"))
cid = os.getenv("OFFICE_CAMERA_ID","office")
cam = [c for c in cams if c.get("id")==cid][0]

conf = float(os.getenv("DETECT_CONF","0.45"))  # можно менять в env
nms = float(os.getenv("DETECT_NMS","0.25"))
img_size = 640
topk = 150

frame = grab_frame_ffmpeg(cam["rtsp_url"])
orig = frame.copy()

masked = frame.copy()
apply_office_mask_2x3(masked)

H, W = masked.shape[:2]

net = cv2.dnn.readNetFromONNX("models/yolov8n.onnx")
inp, r, left, top = letterbox(masked, img_size)
blob = cv2.dnn.blobFromImage(inp, 1/255.0, (img_size, img_size), swapRB=True, crop=False)
net.setInput(blob)
out = net.forward()
if isinstance(out, (list,tuple)):
    out = out[0]
pred = out[0].transpose(1,0)  # (8400,84)

PERSON = 0
boxes=[]
scores=[]

min_w = max(10, int(W*0.03))
min_h = max(14, int(H*0.04))
min_area = int(W*H*0.006)

for row in pred:
    cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    score = float(_sigmoid(row[4:])[PERSON])
    if score < conf:
        continue
    if cx <= 1.5 and cy <= 1.5 and bw <= 1.5 and bh <= 1.5:
        cx *= img_size; cy *= img_size; bw *= img_size; bh *= img_size
    x1 = cx - bw/2; y1 = cy - bh/2; x2 = cx + bw/2; y2 = cy + bh/2
    x1 -= left; x2 -= left; y1 -= top; y2 -= top
    x1 /= r; x2 /= r; y1 /= r; y2 /= r
    x1 = max(0.0, min(W-1.0, x1)); y1 = max(0.0, min(H-1.0, y1))
    x2 = max(0.0, min(W-1.0, x2)); y2 = max(0.0, min(H-1.0, y2))
    if x2<=x1 or y2<=y1:
        continue
    ww=x2-x1; hh=y2-y1
    if ww < min_w or hh < min_h:
        continue
    if (ww*hh) < min_area:
        continue
    if hh < ww*0.65:
        continue
    boxes.append([int(x1), int(y1), int(ww), int(hh)])
    scores.append(score)

if len(scores) > topk:
    order = np.argsort(scores)[::-1][:topk]
    boxes = [boxes[i] for i in order]
    scores = [scores[i] for i in order]

idxs = cv2.dnn.NMSBoxes(boxes, scores, conf, nms)
keep = []
if idxs is not None and len(idxs)>0:
    keep = [int(i) for i in idxs.flatten()]

dbg = masked.copy()
for i in keep:
    x,y,ww,hh = boxes[i]
    cv2.rectangle(dbg, (x,y), (x+ww, y+hh), (0,255,0), 2)
    cv2.putText(dbg, f"person {scores[i]:.2f}", (x, max(15,y-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

ts=int(time.time())
p1=f"photos/yolo_orig_{ts}.jpg"
p2=f"photos/yolo_masked_{ts}.jpg"
p3=f"photos/yolo_boxes_{ts}.jpg"
cv2.imwrite(p1, orig)
cv2.imwrite(p2, masked)
cv2.imwrite(p3, dbg)

print("conf=", conf, "nms=", nms)
print("raw candidates:", len(scores), "kept after NMS:", len(keep))
print("saved:", p1)
print("saved:", p2)
print("saved:", p3)
