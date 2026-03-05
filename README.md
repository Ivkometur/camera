# Camera people detection

RTSP cameras → people detection (YOLO/SSD) → Telegram notifications.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python monitor_office.py
```

## If detection is incorrect

1. Make sure dependencies were installed into the same virtual environment that runs `monitor_office.py`.
2. For the `ultra` detector, verify that `ultralytics` is installed (`python -c "import ultralytics"`).
3. If OpenCV fails to decode RTSP, reinstall OpenCV in the active venv (`pip install -U opencv-python`).
