# Camera people detection

RTSP cameras → people detection (YOLO/SSD) → Telegram notifications.

## Run

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python monitor_office.py
