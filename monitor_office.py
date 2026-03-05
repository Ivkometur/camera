import json
import os
import time
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Optional

import pymysql

from camera_people_yolo import PeopleCounterYOLO
from rtsp_snapshot import rtsp_snapshot_bgr
from tg_notify import tg_send_message


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else str(value).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except Exception:
        return default


def load_cameras() -> List[Dict[str, Any]]:
    raw = env_str("CAMERAS_JSON", "[]")
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def pick_office_camera(cameras: List[Dict[str, Any]]) -> Dict[str, Any]:
    wanted = env_str("OFFICE_CAMERA_ID", "office").lower()

    for camera in cameras:
        if str(camera.get("id", "")).lower() == wanted:
            return camera

    for camera in cameras:
        if str(camera.get("id", "")).lower() == "office":
            return camera

    if cameras:
        return cameras[0]

    raise RuntimeError("No cameras found in CAMERAS_JSON")


def db_connect():
    host = env_str("DB_HOST")
    user = env_str("DB_USER")
    password = env_str("DB_PASSWORD")
    database = env_str("DB_NAME")
    port = env_int("DB_PORT", 3306)

    if not host or not user or not database:
        raise RuntimeError("DB_HOST/DB_USER/DB_NAME must be set")

    return pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        autocommit=True,
        charset="utf8mb4",
    )


def office_workers_count(conn) -> int:
    active_status = env_str("ACTIVE_WORK_STATUS", "work")
    office_task_id = env_int("OFFICE_TASK_ID", 0)
    office_task_name = env_str("OFFICE_WORK_NAME", "Работа в офисе")

    if office_task_id > 0:
        sql = """
        SELECT COUNT(DISTINCT w.usid)
        FROM work w
        WHERE w.work_status = %s
          AND w.task_id = %s
        """
        params = (active_status, office_task_id)
    else:
        sql = """
        SELECT COUNT(DISTINCT w.usid)
        FROM work w
        LEFT JOIN tasks t ON t.task_id = w.task_id
        WHERE w.work_status = %s
          AND t.task_name = %s
        """
        params = (active_status, office_task_name)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if not row:
        return 0

    value = row[0] if isinstance(row, (tuple, list)) else next(iter(row.values()))
    return int(value or 0)


def office_camera_people_count(counter: PeopleCounterYOLO, rtsp_url: str) -> int:
    retries = env_int("SNAPSHOT_RETRIES", 3)
    frames_per_measure = max(3, env_int("CAMERA_COUNT_FRAMES", 7))
    frame_delay_s = max(0.05, env_int("CAMERA_FRAME_DELAY_MS", 200) / 1000.0)

    for _ in range(max(1, retries)):
        try:
            counts: List[int] = []
            for idx in range(frames_per_measure):
                frame = rtsp_snapshot_bgr(rtsp_url)
                result = counter.infer(frame)
                counts.append(int(result.get("count", 0)))

                if idx < frames_per_measure - 1:
                    time.sleep(frame_delay_s)

            if not counts:
                return 0

            # 1) Убираем крайние выбросы (минимум/максимум), если выборка достаточная.
            # 2) По оставшимся значениям берём моду (самое частое), при равенстве — медиану.
            ordered = sorted(counts)
            core = ordered[1:-1] if len(ordered) >= 5 else ordered
            freq = Counter(core)
            max_freq = max(freq.values())
            top = sorted([value for value, amount in freq.items() if amount == max_freq])
            if len(top) == 1:
                return int(top[0])

            median_idx = len(core) // 2
            return int(sorted(core)[median_idx])
        except Exception as exc:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] camera read failed: {exc}", flush=True)
            time.sleep(1)
    raise RuntimeError("Could not read office camera snapshot")


class CountStabilizer:
    def __init__(self):
        self.stable_count: Optional[int] = None
        self.candidate_count: Optional[int] = None
        self.candidate_hits = 0
        self.required_hits = max(1, env_int("COUNT_CONFIRM_LOOPS", 2))

    def update(self, measured_count: int) -> int:
        measured_count = max(0, int(measured_count))

        if self.stable_count is None:
            self.stable_count = measured_count
            self.candidate_count = None
            self.candidate_hits = 0
            return self.stable_count

        if measured_count == self.stable_count:
            self.candidate_count = None
            self.candidate_hits = 0
            return self.stable_count

        if self.candidate_count != measured_count:
            self.candidate_count = measured_count
            self.candidate_hits = 1
            return self.stable_count

        self.candidate_hits += 1
        if self.candidate_hits >= self.required_hits:
            self.stable_count = measured_count
            self.candidate_count = None
            self.candidate_hits = 0

        return self.stable_count


def build_report(office_workers: int, office_camera_people: int) -> str:
    return (
        f"работают в офисе (по БД): {office_workers} чел.\n"
        f"по камере в офисе: {office_camera_people} чел."
    )


def main() -> None:
    admin_chat_id = env_str("ADMIN_CHAT_ID", env_str("ADMIN_TELEGRAM_ID", ""))
    if not admin_chat_id:
        raise SystemExit("ADMIN_CHAT_ID / ADMIN_TELEGRAM_ID is not set")

    interval_s = max(30, env_int("MONITOR_INTERVAL_S", 30))
    model_path = env_str("YOLO_MODEL_PATH", "models/yolov8n.onnx")

    cameras = load_cameras()
    office_camera = pick_office_camera(cameras)
    rtsp_url = str(office_camera.get("rtsp_url", "")).strip()
    if not rtsp_url:
        raise SystemExit("Office camera rtsp_url is empty")

    counter = PeopleCounterYOLO(model_path)
    stabilizer = CountStabilizer()
    conn = db_connect()

    tg_send_message(admin_chat_id, f"✅ Мониторинг запущен. Отправка отчёта каждые {interval_s} секунд.")

    while True:
        try:
            try:
                office_workers = office_workers_count(conn)
            except Exception:
                conn = db_connect()
                office_workers = office_workers_count(conn)

            office_people_measured = office_camera_people_count(counter, rtsp_url)
            office_people = stabilizer.update(office_people_measured)
            message = build_report(office_workers, office_people)

            print(f"[{datetime.now().isoformat(timespec='seconds')}] {message.replace(chr(10), ' | ')}", flush=True)
            tg_send_message(admin_chat_id, message)
        except Exception as exc:
            error_message = f"⚠️ monitor_office error: {exc}"
            print(f"[{datetime.now().isoformat(timespec='seconds')}] {error_message}", flush=True)
            tg_send_message(admin_chat_id, error_message)

        time.sleep(max(1, interval_s))


if __name__ == "__main__":
    main()
