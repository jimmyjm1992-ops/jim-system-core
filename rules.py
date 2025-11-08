# JIM v5 — compact structure rules (HTF drives LTF, break+retest only)
import numpy as np
from indicators import ema, rsi, macd, donchian_high, donchian_low

HTF_TF = "4h"
LTF_TF = "1h"

def classify_trend_4h(closes):
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)
    r   = rsi(closes, 14)
    m, s, h = macd(closes)
    if min(map(len, [e21, e50, r, m])) == 0:
        return "neutral", {"why": "insufficient_data"}
    c = closes[-1]; e21v = e21[-1]; e50v = e50[-1]; rv = r[-1]; mv = m[-1]
    if e21v > e50v and c > e21v and mv > 0 and rv > 50:
        return "bullish", {"c": c, "ema21": e21v, "ema50": e50v, "rsi": rv, "macd": mv}
    if e21v < e50v and c < e21v and mv < 0 and rv < 50:
        return "bearish", {"c": c, "ema21": e21v, "ema50": e50v, "rsi": rv, "macd": mv}
    return "neutral", {"c": c, "ema21": e21v, "ema50": e50v, "rsi": rv, "macd": mv}

def classify_zone_1h(highs, lows, closes, pct=0.3):
    up = donchian_high(highs, 20)
    dn = donchian_low(lows, 20)
    if min(map(len, [up, dn])) == 0:
        return "pending", {"why": "insufficient_data"}
    c = closes[-1]; u = up[-1]; d = dn[-1]
    band = pct / 100.0
    if u and abs((c - u)/u) <= band and c >= u * (1 - band):
        return "breakout_zone", {"c": c, "donchian_high": u}
    if d and abs((c - d)/d) <= band and c <= d * (1 + band):
        return "breakdown_zone", {"c": c, "donchian_low": d}
    return "mid", {"c": c, "donchian_high": u, "donchian_low": d}

def retest_signal(bias4h, highs1h, lows1h, closes1h):
    up = donchian_high(highs1h, 20)
    dn = donchian_low(lows1h, 20)
    if min(map(len, [up, dn])) == 0:
        return {"signal": "none", "note": "insufficient_data"}
    c_prev = closes1h[-2] if len(closes1h) >= 2 else closes1h[-1]
    c_now  = closes1h[-1]
    band = 0.003  # 0.3%

    if bias4h == "bullish":
        if c_prev > up[-2] and abs((c_now - up[-1]) / up[-1]) <= band:
            return {"signal": "LONG",  "entry_zone": [up[-1]*(1-band), up[-1]*(1+band)],
                    "sl_max_pct": 0.012, "note": "breakout + retest band (bull)"}
    if bias4h == "bearish":
        if c_prev < dn[-2] and abs((c_now - dn[-1]) / dn[-1]) <= band:
            return {"signal": "SHORT", "entry_zone": [dn[-1]*(1-band), dn[-1]*(1+band)],
                    "sl_max_pct": 0.012, "note": "breakdown + retest band (bear)"}

    return {"signal": "none", "note": "no retest setup"}
