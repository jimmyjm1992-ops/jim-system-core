# engine.py
import os, time, json, requests
from typing import List

# -------- env / config --------
SUPABASE_URL         = os.getenv("SUPABASE_URL")                 # optional here
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")         # optional here
WORKER_URL           = os.getenv("WORKER_URL") or os.getenv("ALERT_WEBHOOK_URL")  # your Cloudflare Worker URL
PASS_KEY             = os.getenv("PASS_KEY")                     # Worker pass
WATCHLIST            = os.getenv("WATCHLIST",
                                 "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT").split(",")

_loop_period = 60
def set_loop_period(s:int):  # called by app
    global _loop_period
    _loop_period = max(15, int(s))

# -------- helpers --------
def _synthetic_ticket(symbol: str):
    """
    Build a tiny *valid* ticket for your Worker decision() to ACCEPT.
    You can replace this with real OHLCV logic later.
    """
    # make a fake mid and narrow band so SL width < 1.26%
    mid = 100.0
    entry_low, entry_high = mid*0.999, mid*1.001
    sl  = mid*0.988  # ~1.2%
    return {
        "symbol": symbol,
        "side": "LONG",
        "entry": [round(entry_low, 3), round(entry_high, 3)],
        "entry_mid": round(mid, 3),
        "sl": round(sl, 3),
        "tp1": round(mid*1.01, 3),
        "tp2": round(mid*1.02, 3),
        "tp3": round(mid*1.03, 3),
        "macd_slope": "up",
        "rsi": 55.0,
        "atr1h_pct": 0.8,
        "tf": "1h"
    }

def _post_to_worker(ticket: dict):
    if not WORKER_URL or not PASS_KEY:
        print("worker not configured (WORKER_URL/PASS_KEY missing); skipping", flush=True)
        return {"ok": False, "skipped": True}

    url = f"{WORKER_URL}?t={PASS_KEY}"
    r = requests.post(url, json=ticket, timeout=20)
    try:
        data = r.json()
    except Exception:
        data = {"ok": False, "error": f"non-json: {r.text[:200]}", "code": r.status_code}
    print("worker decision:", json.dumps(data)[:300], flush=True)
    return data

# -------- one tick --------
def tick_once():
    """
    One engine iteration:
    - build a ticket per coin (synthetic for now)
    - send to Worker → Worker logs to Supabase & maybe creates order
    """
    t0 = time.strftime("%H:%M:%S")
    print(f"[{t0}] tick start — period { _loop_period }s — {len(WATCHLIST)} symbols", flush=True)

    for sym in WATCHLIST:
        sym = sym.strip().upper()
        if not sym:
            continue
        ticket = _synthetic_ticket(sym)
        resp = _post_to_worker(ticket)
        # Optional: simple routing message based on decision
        if resp.get("ok") and resp.get("state") == "GO":
            print(f"→ {sym}: GO — order_id={resp.get('order_id')}", flush=True)
        else:
            print(f"→ {sym}: WAIT — {resp.get('note')}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] tick end", flush=True)
