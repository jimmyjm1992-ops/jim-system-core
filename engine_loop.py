# engine_loop.py
import time, os, json, requests
from engine import tick_once

PASS_KEY = os.getenv("PASS_KEY")  # same key you use in Cloudflare
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data, timeout=5)
    except:
        pass

def run_once():
    result = tick_once()
    try:
        raw = result.get("raw", [])
        signals = [s for s in raw if s.get("signal") and s.get("signal") != "none"]
        if len(signals) > 0:
            msg = f"🚨 JIM Signal Alert\n\n{json.dumps(signals, indent=2)}"
            send_telegram(msg)
    except Exception as e:
        send_telegram(f"⚠️ JIM Error: {e}")

if __name__ == "__main__":
    while True:
        run_once()
        time.sleep(60)
