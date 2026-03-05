import os
import time
import json
import re
from typing import Any, Dict, Optional, Tuple

import requests

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
ADMIN_CHAT_ID = str(os.getenv("ADMIN_CHAT_ID") or os.getenv("ADMIN_TELEGRAM_ID") or "").strip()

STATE_DIR = (os.getenv("STATE_DIR") or "/opt/factory-bot/state").strip() or "/opt/factory-bot/state"
OFFICE_STATUS_FLAG = os.path.join(STATE_DIR, "office_status_on")
OFFSET_FILE = os.path.join(STATE_DIR, "tg_offset.json")

LAST_TICK_FILE = os.path.join(STATE_DIR, "office_last_tick.json")     # written by monitor_office.py each tick
LAST_TICK_FALLBACK_FILE = os.path.join(STATE_DIR, "office_last_tick.prev.json")
BIAS_FILE = os.path.join(STATE_DIR, "office_bias.json")               # {"mean":float,"n":int,"updated_ts":int}
CALIB_LOG = os.path.join(STATE_DIR, "office_calib.jsonl")             # append-only log

POLL_SEC = float(os.getenv("TG_POLL_SEC", "1.0"))
ALPHA = float(os.getenv("BIAS_ALPHA", "0.30"))
BIAS_MAX_ABS = int(os.getenv("BIAS_MAX_ABS", "6"))
TTL_SEC = int(os.getenv("CALIBRATION_TTL_SEC", "120"))

def _ensure_state_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)

def _load_offset() -> int:
    try:
        with open(OFFSET_FILE, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        return int(d.get("offset", 0))
    except Exception:
        return 0

def _save_offset(offset: int) -> None:
    try:
        tmp = OFFSET_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"offset": int(offset)}, f)
        os.replace(tmp, OFFSET_FILE)
    except Exception:
        pass

def _tg_send(chat_id: str, text: str) -> None:
    chat_id = str(chat_id or "").strip()
    if not BOT_TOKEN or not chat_id:
        print(f"[CTRL] send skipped token={bool(BOT_TOKEN)} chat_id={chat_id!r}", flush=True)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=20)
        if r.status_code != 200:
            print(f"[CTRL] sendMessage error {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[CTRL] sendMessage exception: {e}", flush=True)

def _set_flag(enabled: bool) -> None:
    _ensure_state_dir()
    if enabled:
        try:
            with open(OFFICE_STATUS_FLAG, "w", encoding="utf-8") as f:
                f.write("1\n")
        except Exception:
            pass
    else:
        try:
            if os.path.exists(OFFICE_STATUS_FLAG):
                os.remove(OFFICE_STATUS_FLAG)
        except Exception:
            pass

def _is_admin_chat(chat_id: Any) -> bool:
    try:
        return str(chat_id) == str(ADMIN_CHAT_ID)
    except Exception:
        return False

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return None

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass

def _append_line(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _load_bias() -> Tuple[float, int]:
    d = _read_json(BIAS_FILE) or {}
    try:
        return float(d.get("mean", 0.0)), int(d.get("n", 0))
    except Exception:
        return 0.0, 0

def _update_bias(new_bias: int) -> Tuple[float, int]:
    mean, n = _load_bias()
    # EMA: mean = (1-alpha)*mean + alpha*new_bias
    mean2 = (1.0 - ALPHA) * float(mean) + ALPHA * float(new_bias)
    n2 = int(n) + 1
    _write_json(BIAS_FILE, {"mean": mean2, "n": n2, "updated_ts": int(time.time())})
    return mean2, n2

def _try_apply_calibration(real_count: int) -> Optional[str]:
    now = int(time.time())
    tick = _read_json(LAST_TICK_FILE) or _read_json(LAST_TICK_FALLBACK_FILE)
    if not tick:
        return (
            "Нет последнего замера (office_last_tick.json отсутствует). "
            f"Проверь, что monitor_office.py и tg_control_bot.py используют один STATE_DIR={STATE_DIR}."
        )
    try:
        ts = int(tick.get("ts", 0))
        detected = int(tick.get("detected", 0))
        expected = int(tick.get("expected", 0))
        engine = str(tick.get("engine", ""))
        if now - ts > TTL_SEC:
            return f"Слишком поздно: последний замер был {now-ts}с назад (TTL={TTL_SEC}с). Дождись нового сообщения и отправь число сразу."
    except Exception:
        return "Ошибка чтения последнего замера (office_last_tick.json повреждён)."

    bias = int(real_count) - int(detected)
    if abs(bias) > BIAS_MAX_ABS:
        _append_line(CALIB_LOG, {"ts": now, "real": real_count, "detected": detected, "expected": expected, "bias": bias, "accepted": False, "reason": "bias_too_large"})
        return f"Калибровку НЕ принял: разница {bias} слишком большая (лимит +/-{BIAS_MAX_ABS})."

    mean2, n2 = _update_bias(bias)
    _append_line(CALIB_LOG, {"ts": now, "real": real_count, "detected": detected, "expected": expected, "bias": bias, "accepted": True, "mean": mean2, "n": n2, "engine": engine})
    return f"✅ Калибровка принята: детектор={detected}, реально={real_count}, bias={bias}. Текущий mean_bias={mean2:.2f} (n={n2})"

def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    if not ADMIN_CHAT_ID:
        raise SystemExit("ADMIN_CHAT_ID / ADMIN_TELEGRAM_ID is not set")

    _ensure_state_dir()
    offset = _load_offset()

    _tg_send(ADMIN_CHAT_ID, "🤖 TG control bot started.\nКоманды: /o1 (вкл статус), /o0 (выкл статус), /obias (показать bias)\nКалибровка: после сообщения 'В офисе - X людей' просто пришли правильное число (например 4).")

    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {"timeout": 25, "offset": offset, "allowed_updates": ["message"]}
            r = requests.get(url, params=params, timeout=35)
            if r.status_code != 200:
                print(f"[CTRL] getUpdates error {r.status_code}: {r.text}", flush=True)
                time.sleep(2)
                continue

            data: Dict[str, Any] = r.json() or {}
            if not bool(data.get("ok")):
                print(f"[CTRL] getUpdates not ok: {data}", flush=True)
                time.sleep(2)
                continue

            updates = data.get("result") or []
            for upd in updates:
                try:
                    upd_id = int(upd.get("update_id", 0))
                    offset = max(offset, upd_id + 1)

                    msg = upd.get("message") or {}
                    chat = msg.get("chat") or {}
                    chat_id = chat.get("id")
                    text = (msg.get("text") or "").strip()

                    if not text:
                        continue
                    if not _is_admin_chat(chat_id):
                        continue

                    if text.startswith("/o1"):
                        _set_flag(True)
                        _tg_send(ADMIN_CHAT_ID, "✅ /o1 включён: будет слать 'В офисе - X людей' каждый тик (раз в MONITOR_INTERVAL_S). Если неверно — просто отправь правильное число.")
                        continue

                    if text.startswith("/o0"):
                        _set_flag(False)
                        _tg_send(ADMIN_CHAT_ID, "🛑 /o0 выключен: перестал слать статус каждый тик.")
                        continue

                    if text.startswith("/obias"):
                        mean, n = _load_bias()
                        _tg_send(ADMIN_CHAT_ID, f"ℹ️ mean_bias={mean:.2f} (n={n}), alpha={ALPHA}, ttl={TTL_SEC}s, max_abs={BIAS_MAX_ABS}")
                        continue

                    # calibration: numeric message
                    # accept "4", " 4 ", "+4", "-1", "реально 4"
                    m = re.search(r"[+-]?\d+", text)
                    if m:
                        real = int(m.group(0))
                        if real < 0:
                            _tg_send(ADMIN_CHAT_ID, "Калибровку не применил: число людей не может быть отрицательным.")
                            continue
                        resp = _try_apply_calibration(real)
                        if resp:
                            _tg_send(ADMIN_CHAT_ID, resp)
                        continue

                except Exception as e:
                    print(f"[CTRL] update handle error: {e}", flush=True)

            _save_offset(offset)
            time.sleep(POLL_SEC)

        except Exception as e:
            print(f"[CTRL] loop error: {e}", flush=True)
            time.sleep(2)

if __name__ == "__main__":
    main()
