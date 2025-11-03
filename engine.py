# engine.py
# Pulls data from MEXC (public), applies JIM rules, returns clean objects

import os, time
import ccxt
import numpy as np

from rules import classify_trend_4h, classify_zone_1h, retest_signal

WATCH = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"
]

def _exchange():
    # public-only ok; keys not required for OHLCV
    ex = ccxt.mexc({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    return ex

def _ohlcv(ex, symbol, tf, limit=200):
    # returns (ts, open, high, low, close, vol) arrays
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    arr = np.array(ohlcv, dtype=float)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No OHLCV for {symbol} {tf}")
    return (arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5])

def analyze_symbol(ex, symbol):
    # HTF = 4h, LTF = 1h
    try:
        _, o4, h4, l4, c4, v4 = _ohlcv(ex, symbol, "4h", 220)
        _, o1, h1, l1, c1, v1 = _ohlcv(ex, symbol, "1h", 220)
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "status": "skip"}

    bias4h, bias_meta = classify_trend_4h(c4)
    zone1h, zone_meta = classify_zone_1h(h1, l1, c1, pct=0.3)
    retest = retest_signal(bias4h, h1, l1, c1)

    price = c1[-1]
    out = {
        "symbol": symbol,
        "price": round(float(price), 5),
        "bias": bias4h,
        "zone": zone1h,
        "signal": retest.get("signal", "none"),
        "entry_zone": retest.get("entry_zone"),
        "sl_max_pct": retest.get("sl_max_pct"),
        "note": retest.get("note"),
        "why": {
            "bias": bias_meta,
            "zone": zone_meta
        }
    }
    return out

def tick_once():
    ex = _exchange()
    results = []
    signals = []

    for sym in WATCH:
        r = analyze_symbol(ex, sym)
        results.append(r)
        if r.get("signal") in ("LONG", "SHORT"):
            signals.append(r)

    # (Optional) send signals to your Worker webhook
    # url = os.getenv("ALERT_WEBHOOK_URL")
    # if url and len(signals) > 0:
    #     try:
        #   requests.post(url, json={"kind":"signals", "signals": signals})
    #     except Exception:
    #         pass

    return {
        "engine": "jim_v5_live",
        "scanned": len(WATCH),
        "signals": signals,
        "raw": results,
        "note": "HTF→LTF + break+retest logic active"
    }
