import os
import time
import requests

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
ADMIN_CHAT_ID = str(os.getenv("ADMIN_CHAT_ID") or os.getenv("ADMIN_TELEGRAM_ID") or "").strip()

# Telegram can be slow on VPS -> use longer timeouts + retries
# timeout=(connect_timeout, read_timeout)
TG_TIMEOUT_MSG = (15, 120)
TG_TIMEOUT_PHOTO = (20, 180)

TG_RETRIES = 3
TG_RETRY_SLEEP_S = 2.0

# If photo is too large, try to downscale/compress before sending
TG_PHOTO_MAX_BYTES = int(os.getenv("TG_PHOTO_MAX_BYTES", "700000"))  # ~0.7MB
TG_PHOTO_MAX_W = int(os.getenv("TG_PHOTO_MAX_W", "1280"))          # downscale width
TG_PHOTO_JPEG_QUALITY = int(os.getenv("TG_PHOTO_JPEG_QUALITY", "70"))


def _tg_url(method: str) -> str:
    return f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"


def tg_send_message(chat_id: str, text: str) -> None:
    chat_id = str(chat_id or "").strip()

    if not BOT_TOKEN or not chat_id:
        print(f"[TG] message skipped token={bool(BOT_TOKEN)} chat_id={chat_id!r}", flush=True)
        return

    url = _tg_url("sendMessage")

    last_err = None
    for attempt in range(1, TG_RETRIES + 1):
        try:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "text": text},
                timeout=TG_TIMEOUT_MSG,
            )
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}: {r.text}"
            print(f"[TG] sendMessage error attempt={attempt}/{TG_RETRIES}: {last_err}", flush=True)
        except Exception as e:
            last_err = str(e)
            print(f"[TG] sendMessage exception attempt={attempt}/{TG_RETRIES}: {e}", flush=True)

        time.sleep(TG_RETRY_SLEEP_S)

    print(f"[TG] sendMessage FAILED after retries: {last_err}", flush=True)


def _maybe_downscale_jpeg(in_path: str) -> str:
    """
    If file is big -> try to create smaller jpeg next to it:
      <in_path>.tg.jpg
    Returns path to send (original or resized).
    """
    try:
        if not os.path.exists(in_path):
            return in_path

        size = os.path.getsize(in_path)
        if size <= TG_PHOTO_MAX_BYTES:
            return in_path

        # try cv2 if available
        try:
            import cv2  # type: ignore
        except Exception:
            print(f"[TG] photo too large ({size}) but cv2 not available -> send original", flush=True)
            return in_path

        img = cv2.imread(in_path)
        if img is None:
            return in_path

        h, w = img.shape[:2]
        if w > TG_PHOTO_MAX_W:
            scale = TG_PHOTO_MAX_W / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_path = in_path + ".tg.jpg"
        ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(TG_PHOTO_JPEG_QUALITY)])
        if ok and os.path.exists(out_path):
            out_size = os.path.getsize(out_path)
            print(f"[TG] downscaled photo: {in_path} {size}B -> {out_path} {out_size}B", flush=True)
            return out_path

        return in_path

    except Exception as e:
        print(f"[TG] downscale failed: {e}", flush=True)
        return in_path


def tg_send_photo(chat_id: str, photo_path: str, caption: str = "") -> None:
    chat_id = str(chat_id or "").strip()

    if not BOT_TOKEN or not chat_id:
        print(f"[TG] photo skipped token={bool(BOT_TOKEN)} chat_id={chat_id!r}", flush=True)
        return

    if not os.path.exists(photo_path):
        print(f"[TG] photo file not found: {photo_path}", flush=True)
        return

    # optionally shrink/compress
    send_path = _maybe_downscale_jpeg(photo_path)

    url = _tg_url("sendPhoto")

    last_err = None
    for attempt in range(1, TG_RETRIES + 1):
        try:
            size = os.path.getsize(send_path)
            print(f"[TG] sending photo attempt={attempt}/{TG_RETRIES} path={send_path} size={size}", flush=True)

            with open(send_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption

                r = requests.post(
                    url,
                    data=data,
                    files=files,
                    timeout=TG_TIMEOUT_PHOTO,
                )

            if r.status_code == 200:
                print("[TG] photo sent OK", flush=True)
                return

            last_err = f"HTTP {r.status_code}: {r.text}"
            print(f"[TG] sendPhoto error attempt={attempt}/{TG_RETRIES}: {last_err}", flush=True)

        except Exception as e:
            last_err = str(e)
            print(f"[TG] sendPhoto exception attempt={attempt}/{TG_RETRIES}: {e}", flush=True)

        time.sleep(TG_RETRY_SLEEP_S)

    print(f"[TG] sendPhoto FAILED after retries: {last_err}", flush=True)
