import os
import time
import json
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pymysql

from camera_people_yolo import PeopleCounterYOLO
from rtsp_snapshot import rtsp_snapshot_bgr
from tg_notify import tg_send_message, tg_send_photo

# --- /o1 status mode flag (controlled by tg_control_bot.py) ---
def _office_status_enabled() -> bool:
    try:
        state_dir = (os.getenv("STATE_DIR") or "/opt/factory-bot/state").strip() or "/opt/factory-bot/state"
        return os.path.exists(os.path.join(state_dir, "office_status_on"))
    except Exception:
        return False


# --- Optional ULTRA detector (won't crash if module missing) ---
try:
    from camera_people_ultra import PeopleCounterUltra  # type: ignore
except Exception:
    PeopleCounterUltra = None  # type: ignore


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# --- state flags + calibration (controlled by tg_control_bot.py) ---
STATE_DIR = _env_str("STATE_DIR", "/opt/factory-bot/state").strip() or "/opt/factory-bot/state"
OFFICE_STATUS_FLAG = os.path.join(STATE_DIR, "office_status_on")
OFFICE_LAST_TICK_FILE = os.path.join(STATE_DIR, "office_last_tick.json")
OFFICE_BIAS_FILE = os.path.join(STATE_DIR, "office_bias.json")

BIAS_ALPHA = _env_float("BIAS_ALPHA", 0.30)
BIAS_MAX_ABS = _env_int("BIAS_MAX_ABS", 6)

def _office_status_enabled() -> bool:
    try:
        return os.path.exists(OFFICE_STATUS_FLAG)
    except Exception:
        return False

def _load_bias_mean() -> float:
    try:
        with open(OFFICE_BIAS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        return float(d.get("mean", 0.0))
    except Exception:
        return 0.0

def _apply_bias(detected: int) -> int:
    # Use rounded mean bias, clamp to non-negative
    mean = _load_bias_mean()
    adj = int(round(float(detected) + float(mean)))
    return max(0, adj)

def _write_last_tick(ts: int, detected: int, expected: int, engine: str) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp = OFFICE_LAST_TICK_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ts": int(ts), "detected": int(detected), "expected": int(expected), "engine": str(engine)}, f, ensure_ascii=False)
        os.replace(tmp, OFFICE_LAST_TICK_FILE)
    except Exception:
        pass

def db_connect():
    host = _env_str("DB_HOST")
    user = _env_str("DB_USER")
    password = _env_str("DB_PASSWORD")
    dbname = _env_str("DB_NAME")
    port = _env_int("DB_PORT", 3306)

    if not host or not user or not dbname:
        raise SystemExit("DB_HOST/DB_USER/DB_NAME not set in .env")

    return pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=dbname,
        port=port,
        autocommit=True,
    )


def load_cameras() -> List[Dict[str, Any]]:
    cams_json = _env_str("CAMERAS_JSON", "[]").strip()
    try:
        cams = json.loads(cams_json)
        if not isinstance(cams, list):
            return []
        return cams
    except Exception:
        return []


def pick_office_camera(cams: List[Dict[str, Any]]) -> Dict[str, Any]:
    want_id = _env_str("OFFICE_CAMERA_ID", "office").strip().lower()

    for c in cams:
        if str(c.get("id", "")).strip().lower() == want_id:
            return c

    for c in cams:
        if str(c.get("id", "")).strip().lower() == "office":
            return c

    if cams:
        return cams[0]

    raise SystemExit("No cameras in CAMERAS_JSON")


def get_expected_office(conn) -> tuple[int, list[str], list[tuple]]:
    """
    Returns:
      expected: int
      names: list[str]
      raw_rows: list[tuple]  (work_id, fio, uname, task_name, accept_dt)

    We detect "office work" by either:
      - OFFICE_TASK_ID (preferred, numeric)
      - OR tasks.task_name == OFFICE_WORK_NAME
    """
    office_task_id = _env_int("OFFICE_TASK_ID", 0)
    office_name = _env_str("OFFICE_WORK_NAME", "Работа в офисе").strip()

    with conn.cursor() as cur:
        if office_task_id > 0:
            cur.execute(
                """
                SELECT
                  w.work_id,
                  u.fio,
                  u.uname,
                  t.task_name,
                  w.accept_dt
                FROM work w
                LEFT JOIN users u ON u.usid = w.usid
                LEFT JOIN tasks t ON t.task_id = w.task_id
                WHERE w.work_status=%s
                  AND w.task_id=%s
                ORDER BY w.accept_dt DESC
                """,
                ("work", office_task_id),
            )
        else:
            cur.execute(
                """
                SELECT
                  w.work_id,
                  u.fio,
                  u.uname,
                  t.task_name,
                  w.accept_dt
                FROM work w
                LEFT JOIN users u ON u.usid = w.usid
                LEFT JOIN tasks t ON t.task_id = w.task_id
                WHERE w.work_status=%s
                  AND t.task_name=%s
                ORDER BY w.accept_dt DESC
                """,
                ("work", office_name),
            )
        rows = cur.fetchall() or []

    def _name(fio, uname, work_id):
        fio = (fio or "").strip()
        uname = (uname or "").strip()
        return fio or uname or f"work_id={work_id}"

    names = [_name(r[1], r[2], r[0]) for r in rows]
    expected = len(rows)
    return expected, names, list(rows)


def fmt_stats(stats: Dict[str, Any]) -> str:
    w, h = stats.get("frame_wh", ("?", "?"))
    return (
        f"frame={w}x{h} imgsz={stats.get('imgsz')} conf={stats.get('conf')} iou={stats.get('iou')} "
        f"t_ms={stats.get('t_ms')}"
    )


def _fmt_dur(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    mm = seconds // 60
    ss = seconds % 60
    if mm <= 0:
        return f"{ss}с"
    return f"{mm}м {ss}с"


def _save_debug_photo(counter, frame, res, prefix: str) -> Optional[str]:
    """
    Try to save debug photo with boxes. Returns path or None.
    Important: if res['masked'] is None -> pass original frame to draw_debug.
    """
    try:
        dets = res.get("detections") or []
        masked = res.get("masked", None)
        if masked is None:
            masked = frame
        paths = counter.draw_debug(frame, masked, dets, prefix)
        return paths.get("boxes")
    except Exception as e:
        print(f"[WARN] debug photo save failed: {e}", flush=True)
        return None


def people_count_window(
    counter,
    rtsp_url: str,
    n: int,
    sleep_s: float,
    log_each: bool,
) -> Tuple[int, List[int], Dict[str, Any], Any, Any]:
    """
    Take n snapshots quickly, run detector.
    Returns:
      median, counts, last_stats, last_res, last_frame
    """
    counts: List[int] = []
    last_stats: Dict[str, Any] = {}
    last_res: Any = None
    last_frame: Any = None

    for i in range(n):
        frame = rtsp_snapshot_bgr(rtsp_url)
        res = counter.infer(frame)

        cnt = int(res.get("count", 0))
        counts.append(cnt)

        last_stats = res.get("stats") or {}
        last_stats["frame_wh"] = last_stats.get("frame_wh") or (frame.shape[1], frame.shape[0])

        last_res = res
        last_frame = frame

        if log_each:
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] snap={i+1}/{n} cnt={cnt} {fmt_stats(last_stats)}",
                flush=True,
            )

        time.sleep(max(0.0, sleep_s))

    med = int(statistics.median(counts)) if counts else 0
    return med, counts, last_stats, last_res, last_frame


def main():
    model_path = _env_str("YOLO_MODEL_PATH", "models/yolov8n.onnx")
    engine = _env_str("DETECTOR_ENGINE", "opencv").strip().lower()

    admin_chat_id = _env_str("ADMIN_CHAT_ID", _env_str("ADMIN_TELEGRAM_ID", "")).strip()
    if not admin_chat_id:
        raise SystemExit("ADMIN_CHAT_ID / ADMIN_TELEGRAM_ID not set")

    cams = load_cameras()
    cam = pick_office_camera(cams)
    rtsp_url = str(cam.get("rtsp_url", "")).strip()
    room = str(cam.get("room", "Office")).strip()
    if not rtsp_url:
        raise SystemExit("office camera rtsp_url is empty")

    interval_s = _env_int("MONITOR_INTERVAL_S", _env_int("MONITOR_INTERVAL_SEC", 10))

    # window settings
    window_n = _env_int("WINDOW_N", 5)
    window_sleep_s = _env_float("WINDOW_SLEEP_S", 0.12)
    log_each_snapshot = _env_bool("LOG_EACH_SNAPSHOT", False)

    # anti-flap: accept pair change only if seen N times подряд
    pair_stable_n = _env_int("PAIR_STABLE_N", 3)

    # notifications toggles
    report_drop = _env_bool("REPORT_DROP_IMMEDIATELY", True)
    drop_min_delta = _env_int("DROP_MIN_DELTA", 1)
    notify_mismatch = _env_bool("REPORT_MISMATCH", True)

    # IMPORTANT: mismatch anti-spam policy
    # send mismatch ONLY ON ENTER ("ok"->less/more) and ONLY ON RECOVER ("less/more"->ok)
    mismatch_send_only_on_transition = _env_bool("MISMATCH_ONLY_ON_TRANSITION", True)

    # photo settings (saved ONLY on events)
    debug_save_photos = _env_bool("DEBUG_SAVE_PHOTOS", True)

    # Detector (ONE selection)
    if engine == "ultra" and PeopleCounterUltra is not None:
        counter = PeopleCounterUltra()
    else:
        if engine == "ultra" and PeopleCounterUltra is None:
            print("[WARN] DETECTOR_ENGINE=ultra but camera_people_ultra.py missing -> fallback to PeopleCounterYOLO", flush=True)
        counter = PeopleCounterYOLO(model_path)

    conn = db_connect()

    # --- state ---
    prev_detected: Optional[int] = None

    # pair debouncer
    last_pair_effective: Optional[Tuple[int, int]] = None
    pending_pair: Optional[Tuple[int, int]] = None
    pending_count: int = 0

    # mismatch episode
    mismatch_state: Optional[str] = None  # "less" | "more" | None
    mismatch_since: Optional[datetime] = None

    tg_send_message(
        admin_chat_id,
        f"✅ factory-monitor started. room={room}. interval={interval_s}s engine={engine} "
        f"window_n={window_n} stable_n={pair_stable_n}",
    )

    while True:
        try:
            expected, expected_names, _rows = get_expected_office(conn)

            detected, counts, stats, last_res, last_frame = people_count_window(
                counter,
                rtsp_url,
                n=window_n,
                sleep_s=window_sleep_s,
                log_each=log_each_snapshot,
            )

            stats["counts_window"] = counts
            stats["median"] = detected

            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] expected={expected} detected={detected} "
                f"win={counts} median={detected} {fmt_stats(stats)}",
                flush=True,
            )

            # keep latest measurement for tg_control_bot calibration replies
            _write_last_tick(int(time.time()), detected, expected, engine)

            ### [O1_STATUS_SEND] ###
            # /o1 enabled -> send short status each tick
            try:
                if _office_status_enabled():
                    tg_send_message(admin_chat_id, f"В офисе - {detected} людей")
            except Exception as _e:
                print(f"[WARN] /o1 status send failed: {_e}", flush=True)
### [O1_STATUS_SEND] ###

            raw_pair = (expected, detected)

            # ---- pair debouncer (anti-flap) ----
            if pending_pair != raw_pair:
                pending_pair = raw_pair
                pending_count = 1
            else:
                pending_count += 1

            # accept effective pair only after stable_n repeats
            pair_effective = last_pair_effective
            if pending_pair is not None and pending_count >= max(1, pair_stable_n):
                pair_effective = pending_pair

            # if effective pair changed -> EVENT
            if pair_effective != last_pair_effective and pair_effective is not None:
                now = datetime.now()
                expected_eff, detected_eff = pair_effective

                names_txt = ", ".join(expected_names) if expected_names else "-"

                # save one debug photo ONLY FOR EVENT
                debug_boxes_path: Optional[str] = None
                if debug_save_photos and last_frame is not None and last_res is not None:
                    ts2 = int(time.time())
                    prefix = f"/opt/factory-bot/photos/event_{ts2}"
                    debug_boxes_path = _save_debug_photo(counter, last_frame, last_res, prefix)

                # 1) Drop event (camera decreased vs previous tick)
                if report_drop and prev_detected is not None:
                    if detected_eff <= (prev_detected - drop_min_delta):
                        msg = (
                            f"📉 Уменьшилось количество людей на камере\n"
                            f"Комната: {room}\n"
                            f"Было: {prev_detected}\n"
                            f"Стало (median/{window_n}): {detected_eff}\n"
                            f"Ожидается (Работа в офисе): {expected_eff}\n"
                        )
                        if expected_names:
                            msg += "Имена (по БД): " + ", ".join(expected_names) + "\n"
                        msg += "Stats: " + fmt_stats(stats)
                        tg_send_message(admin_chat_id, msg)
                        if debug_boxes_path:
                            tg_send_photo(admin_chat_id, debug_boxes_path, caption="Drop (boxes)")

                # 2) Mismatch / recovered (NO SPAM)
                if notify_mismatch:
                    if detected_eff < expected_eff:
                        state = "less"
                    elif detected_eff > expected_eff:
                        state = "more"
                    else:
                        state = None

                    if mismatch_send_only_on_transition:
                        # recovered
                        if state is None and mismatch_state is not None:
                            dur_s = int((now - (mismatch_since or now)).total_seconds())
                            msg = (
                                f"✅ Всё отлично — совпало\n"
                                f"Комната: {room}\n"
                                f"Ожидается: {expected_eff}\n"
                                f"На камере: {detected_eff}\n"
                                f"Несоответствие длилось: {_fmt_dur(dur_s)}\n"
                                f"Stats: {fmt_stats(stats)}"
                            )
                            tg_send_message(admin_chat_id, msg)
                            if debug_boxes_path:
                                tg_send_photo(admin_chat_id, debug_boxes_path, caption="Recovered (boxes)")
                            mismatch_state = None
                            mismatch_since = None

                        # entering mismatch
                        elif state is not None and mismatch_state is None:
                            mismatch_state = state
                            mismatch_since = now
                            head = "🚨 Несоответствие: людей МЕНЬШЕ нормы" if state == "less" else "🚨 Несоответствие: людей БОЛЬШЕ нормы"
                            msg = (
                                f"{head}\n"
                                f"Комната: {room}\n"
                                f"Ожидается (Работа в офисе): {expected_eff}\n"
                                f"Имена (по БД): {names_txt}\n"
                                f"На камере (median/{window_n}): {detected_eff}\n"
                                f"Diff (expected-detected): {expected_eff - detected_eff}\n"
                                f"Stats: {fmt_stats(stats)}"
                            )
                            tg_send_message(admin_chat_id, msg)
                            if debug_boxes_path:
                                tg_send_photo(admin_chat_id, debug_boxes_path, caption="Mismatch (boxes)")

                        # switch less<->more
                        elif state is not None and mismatch_state is not None and state != mismatch_state:
                            mismatch_state = state
                            mismatch_since = now
                            head = "🚨 Несоответствие: людей МЕНЬШЕ нормы" if state == "less" else "🚨 Несоответствие: людей БОЛЬШЕ нормы"
                            msg = (
                                f"{head}\n"
                                f"Комната: {room}\n"
                                f"Ожидается (Работа в офисе): {expected_eff}\n"
                                f"Имена (по БД): {names_txt}\n"
                                f"На камере (median/{window_n}): {detected_eff}\n"
                                f"Diff (expected-detected): {expected_eff - detected_eff}\n"
                                f"Stats: {fmt_stats(stats)}"
                            )
                            tg_send_message(admin_chat_id, msg)
                            if debug_boxes_path:
                                tg_send_photo(admin_chat_id, debug_boxes_path, caption="Mismatch (boxes)")

                last_pair_effective = pair_effective

            # always update prev_detected (drop baseline)
            prev_detected = detected

            time.sleep(max(1, interval_s))

        except Exception as e:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] ERROR: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
