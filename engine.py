# engine.py — JIM v5.9.2 (drop-in upgrade)
# - Body-close breakout detect
# - Dynamic ATR retest bands (Normal/Momentum/News)
# - Momentum score (MACD slope + RSI bands)
# - Body-break strength %
# - Optional volume slope check
# - Same output shape + same POST to Worker /signals?t=PASS

import os, json, math, pathlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt
import httpx
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

load_dotenv()

# ---------- Env ----------
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").rstrip("/")   # e.g. https://<worker>/signals?t=PASS
PASS_KEY          = os.getenv("PASS_KEY", "")
EXCHANGE_NAME     = os.getenv("PRICE_EXCHANGE", "mexc")
WATCHLIST         = [s.strip() for s in os.getenv("WATCHLIST","BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if s.strip()]
MAX_SL_PCT        = float(os.getenv("MAX_SL_PCT", "1.26")) / 100.0    # 1.26% default
USE_VOL_SLOPE     = os.getenv("USE_VOL_SLOPE", "0") == "1"            # optional

# ---------- CCXT helpers ----------
def _exchange():
    cls = getattr(ccxt, EXCHANGE_NAME.lower())
    return cls({"enableRateLimit": True})

def _mk(symbol):
    return symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"

def _ohlcv_to_df(kl):
    df = pd.DataFrame(kl, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df.astype(float)

def fetch_tf(ex, sym, tf="1h", limit=400):
    return _ohlcv_to_df(ex.fetch_ohlcv(_mk(sym), timeframe=tf, limit=limit))

# ---------- Indicators ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    sigl = ema(line, sig)
    hist = line - sigl
    return line, sigl, hist

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0).rolling(n).mean()
    dn = (-d.clip(upper=0.0)).rolling(n).mean().replace(0, np.nan)
    rs = up / dn
    out = 100 - (100/(1+rs))
    return out.bfill().ffill().fillna(50.0)

def atr(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def donchian(df, n=55):
    # use previous fully closed 4H candle as anchor
    cap  = df["high"].rolling(n).max().iloc[-2]
    base = df["low"].rolling(n).min().iloc[-2]
    return cap, base

def slope(series):
    return "up" if series.iloc[-1] > series.iloc[-2] else ("down" if series.iloc[-1] < series.iloc[-2] else "flat")

# ---------- Core analyze (v5.9.2) ----------
def analyze_symbol(ex, symbol):
    # HTF anchor on 4H, execution on 1H
    d4h = fetch_tf(ex, symbol, "4h", 300)
    d1h = fetch_tf(ex, symbol, "1h", 500)

    # HTF trend + MACD slope
    d4h["ema21"] = ema(d4h.close, 21)
    d4h["ema50"] = ema(d4h.close, 50)
    d4h["macd"], d4h["macdsig"], _ = macd(d4h.close)
    trend4h = "bull" if d4h.ema21.iloc[-1] > d4h.ema50.iloc[-1] else "bear"
    macd4   = slope(d4h["macd"])

    # LTF momentum + ATR + volume
    d1h["ema21"] = ema(d1h.close, 21)
    d1h["ema50"] = ema(d1h.close, 50)
    d1h["macd"], d1h["macdsig"], _ = macd(d1h.close)
    d1h["rsi"]   = rsi(d1h.close, 14)
    d1h["atr"]   = atr(d1h, 14)              # raw ATR (price units)
    d1h["atrpct"]= d1h["atr"] / d1h["close"]  # ATR as %
    if USE_VOL_SLOPE:
        d1h["vol_sma"] = d1h["vol"].rolling(20).mean()

    cap, base = donchian(d4h, 55)
    c = float(d1h.close.iloc[-1])
    prev_c = float(d1h.close.iloc[-2])

    zone = "mid"
    if c >= cap: zone = "above_cap"
    elif c <= base: zone = "below_base"

    # Body-based breakout: confirm body close beyond anchor
    body_break_up   = (prev_c <= cap)  and (c > cap)
    body_break_down = (prev_c >= base) and (c < base)

    # Body-break strength (how far body closed beyond the level)
    body_break_pct = None
    if body_break_up:
        body_break_pct = (c - cap) / cap * 100.0
    if body_break_down:
        body_break_pct = (base - c) / base * 100.0

    # Dynamic retest bands by regime
    atrpct = float(d1h.atrpct.iloc[-1] if pd.notna(d1h.atrpct.iloc[-1]) else 0.006)  # fallback 0.6%
    # NORMAL: ~0.05%–0.30%; MOMENTUM: 0.02%–0.18%; NEWS: 0.10%–0.50%
    band_normal   = (0.0005, 0.0030)
    band_momentum = (0.0002, 0.0018)
    band_news     = (0.0010, 0.0050)

    # Momentum score: MACD slope + RSI band (50–65 long / 35–48 short)
    rsi_now = float(d1h.rsi.iloc[-1])
    macd1   = slope(d1h["macd"])
    mom_score_long  = (1 if macd1 == "up"   else 0) + (1 if 50.0 <= rsi_now <= 65.0 else 0)
    mom_score_short = (1 if macd1 == "down" else 0) + (1 if 35.0 <= rsi_now <= 48.0 else 0)
    use_mom_band = mom_score_long == 2 or mom_score_short == 2

    def band(level, side):
        lo, hi = band_momentum if use_mom_band else band_normal
        # If very high ATR (>1.5%), widen a bit like "news"
        if atrpct > 0.015:
            lo, hi = band_news
        if side == "long":
            # HODL above cap → retest slightly BELOW cap
            return (level * (1 - hi), level * (1 - lo))
        else:
            # SHORT below base → retest slightly ABOVE base
            return (level * (1 + lo), level * (1 + hi))

    signal = "none"
    entry_band = None
    sl_max_pct = None
    note = "no retest setup"

    # Optional volume slope (very soft)
    vol_ok = True
    if USE_VOL_SLOPE and pd.notna(d1h["vol_sma"].iloc[-1]) and pd.notna(d1h["vol_sma"].iloc[-2]):
        vol_ok = d1h["vol_sma"].iloc[-1] >= d1h["vol_sma"].iloc[-2] * 0.98  # rising or flat

    # LONG path: 4H trend bull + MACD up; body close above cap; momentum acceptable
    if body_break_up and trend4h == "bull" and macd4 in ("up", "flat"):
        if mom_score_long >= 1 and vol_ok:
            entry_band = band(cap, "long")
            sl_max_pct = MAX_SL_PCT
            signal = "LONG"
            note = f"Body break above cap ({body_break_pct:.3f}%); retest band below cap"

    # SHORT path: 4H trend bear + MACD down; body close below base; momentum acceptable
    if body_break_down and trend4h == "bear" and macd4 in ("down", "flat"):
        if mom_score_short >= 1 and vol_ok:
            entry_band = band(base, "short")
            sl_max_pct = MAX_SL_PCT
            signal = "SHORT"
            note = f"Body break below base ({body_break_pct:.3f}%); retest band above base"

    out = {
        "symbol": _mk(symbol),
        "price": c,
        "bias": "bullish" if trend4h == "bull" else "bearish",
        "zone": zone,
        "signal": signal,
        "entry_zone": entry_band,
        "sl_max_pct": sl_max_pct,
        "note": note,
        "why": {
            "bias": {
                "c": c,
                "ema21": float(d1h.ema21.iloc[-1]),
                "ema50": float(d1h.ema50.iloc[-1]),
                "rsi": rsi_now,
                "macd": float(d1h.macd.iloc[-1]),
            },
            "zone": {
                "c": c,
                "donchian_high": float(cap),
                "donchian_low": float(base),
            },
            "metrics": {
                "atr1h_pct": atrpct * 100.0,
                "mom_long": mom_score_long,
                "mom_short": mom_score_short,
                "body_break_pct": body_break_pct if body_break_pct is not None else 0.0,
                "macd4_slope": macd4
            }
        },
    }
    return out

# ---------- TICK + optional push ----------
def tick_once():
    ex = _exchange()
    results, signals = [], []

    for sym in WATCHLIST:
        try:
            row = analyze_symbol(ex, sym)
            results.append(row)
            if row["signal"] in ("LONG","SHORT") and row.get("entry_zone"):
                # build compact ticket for Worker/Supabase
                lo, hi = row["entry_zone"]
                ticket = {
                    "symbol": row["symbol"],
                    "side": row["signal"],
                    "entry": [lo, hi],
                    "sl_max_pct": row["sl_max_pct"],
                    "tp1": None, "tp2": None, "tp3": None,       # optional: let GPT/you set targets later
                    "rsi": row["why"]["bias"]["rsi"],
                    "macd_slope": row["why"]["metrics"]["macd4_slope"],
                    "atr1h_pct": row["why"]["metrics"]["atr1h_pct"],
                    "spot": row["price"],
                    "note": row["note"],
                }
                signals.append(ticket)
        except Exception as e:
            results.append({"symbol": _mk(sym), "error": str(e)})

    # Push compact batch to Worker /signals (same style you already use)
    if ALERT_WEBHOOK_URL and signals:
        payload = { "pass": PASS_KEY, "summary": f"{len(signals)} signal(s) v5.9.2", "signals": signals }
        try:
            with httpx.Client(timeout=10.0) as cli:
                cli.post(ALERT_WEBHOOK_URL, json=payload)
        except Exception:
            pass

    return {
        "engine": "jim_v5.9.2",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": results,
        "note": "v5.9.2 structure+momentum engine active",
    }
