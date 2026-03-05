import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import pymysql

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")  # <-- важно: явный путь

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("factory-bot")

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
ADMIN_CHAT_ID = str(os.getenv("ADMIN_TELEGRAM_ID") or "").strip()

DB_HOST = (os.getenv("DB_HOST") or "127.0.0.1").strip()
DB_PORT = int(os.getenv("DB_PORT") or "3306")
DB_NAME = (os.getenv("DB_NAME") or "factory").strip()
DB_USER = (os.getenv("DB_USER") or "").strip()
DB_PASSWORD = (os.getenv("DB_PASSWORD") or "").strip()

ACTIVE_WORK_STATUS = (os.getenv("ACTIVE_WORK_STATUS") or "work").strip()

if not BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
if not DB_USER:
    raise SystemExit("DB_USER is not set")

def is_admin(update: Update) -> bool:
    if not ADMIN_CHAT_ID:
        return True
    try:
        return str(update.effective_chat.id) == str(ADMIN_CHAT_ID)
    except Exception:
        return False

def db_connect():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

def fetch_active_works(limit: int = 200):
    sql = """
    SELECT
      w.work_id,
      w.usid,
      u.uname,
      u.fio,
      w.task_id,
      t.task_name,
      w.work_status,
      w.accept_dt,
      w.created_dt
    FROM work w
    LEFT JOIN users u ON u.usid = w.usid
    LEFT JOIN tasks t ON t.task_id = w.task_id
    WHERE w.work_status = %s
    ORDER BY w.accept_dt DESC
    LIMIT %s
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ACTIVE_WORK_STATUS, limit))
            return cur.fetchall()

def fmt(rows) -> str:
    if not rows:
        return f"Активных работ сейчас нет (work_status='{ACTIVE_WORK_STATUS}')."
    lines = [f"Активные работы (status='{ACTIVE_WORK_STATUS}'): {len(rows)} шт.\n"]
    for r in rows:
        work_id = r.get("work_id")
        task = r.get("task_name") or f"task_id={r.get('task_id')}"
        fio = r.get("fio") or "-"
        uname = r.get("uname") or "-"
        accept_dt = r.get("accept_dt") or "-"
        lines.append(f"• #{work_id} — {task}\n  👤 {fio} ({uname})\n  🕒 {accept_dt}")
    return "\n".join(lines)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text(
        "factory-bot online.\n"
        "Команды:\n"
        "/active — активные работы + кто выполняет\n"
        "/ping — проверка\n"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text("pong")

async def cmd_active(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    try:
        rows = fetch_active_works(limit=200)
        text = fmt(rows)
        if len(text) <= 3800:
            await update.message.reply_text(text)
            return
        chunk = ""
        for line in text.splitlines(True):
            if len(chunk) + len(line) > 3800:
                await update.message.reply_text(chunk)
                chunk = ""
            chunk += line
        if chunk:
            await update.message.reply_text(chunk)
    except Exception as e:
        log.exception("active failed")
        await update.message.reply_text(f"Ошибка: {e}")

def main():
    log.info("factory-bot started (polling)")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("active", cmd_active))
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
