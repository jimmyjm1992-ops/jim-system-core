# engine.py — JIM v5.9.2 radar + trigger scan (single tick)

import os, json, math, time, pathlib
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

# --- ENV ---
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").rstrip("/")
PASS_KEY          = os.getenv("PASS_KEY", "")
EXCHANGE_NAME     = os.getenv("PRICE_EXCHANGE", "binance")
WATCHLIST         = [s.strip() for s in os.getenv("WATCHLIST","BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if s.strip()]

# --- EXCHANGE (spot feed only) ---
def _exchange():
    name = EXCHANGE_NAME.lower()
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True})

def _mk(symbol):
    return symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"

# --- INDICATORS ---
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    sigl = ema(line, sig)
    hist = line - sigl
    return line, sigl, hist
def rsi(close, n=14):
    d = close.diff()
    up = pd.Series(np.where(d>0, d, 0.0), index=close.index).rolling(n).mean()
    dn = pd.Series(np.where(d<0, -d, 0.0), index=close.index).rolling(n).mean().replace(0, np.nan)
    rs = up / dn
    out = 100 - (100/(1+rs))
    return out.bfill().ffill().fillna(50)
def donchian(df, n=55):
    cap  = df["high"].rolling(n).max().iloc[-2]
    base = df["low"].rolling(n).min().iloc[-2]
    return cap, base
def macd_slope(series):
    return "up" if series.iloc[-1] > series.iloc[-2] else ("down" if series.iloc[-1] < series.iloc[-2] else "flat")

def _ohlcv(kl):
    df = pd.DataFrame(kl, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    for c in ["open","high","low","close","vol"]:
        df[c] = df[c].astype(float)
    return df

def fetch_tf(ex, sym, tf="1h", limit=400):
    return _ohlcv(ex.fetch_ohlcv(_mk(sym), timeframe=tf, limit=limit))

# --- CORE ANALYZE ---
def analyze_symbol(ex, symbol):
    d4h = fetch_tf(ex, symbol, "4h", 300)
    d1h = fetch_tf(ex, symbol, "1h", 400)

    d4h["ema21"] = ema(d4h.close, 21)
    d4h["ema50"] = ema(d4h.close, 50)
    d4h["macd"], d4h["macdsig"], _ = macd(d4h.close)
    trend4h = "bull" if d4h.ema21.iloc[-1] > d4h.ema50.iloc[-1] else "bear"
    macd4   = macd_slope(d4h["macd"])

    d1h["ema21"] = ema(d1h.close, 21)
    d1h["ema50"] = ema(d1h.close, 50)
    d1h["macd"], d1h["macdsig"], _ = macd(d1h.close)
    d1h["rsi"]   = rsi(d1h.close, 14)

    cap, base = donchian(d4h, 55)
    c = float(d1h.close.iloc[-1])

    zone = "mid"
    if c >= cap: zone = "above_cap"
    elif c <= base: zone = "below_base"

    signal = "none"
    entry_band = None
    sl_max_pct = None
    note = "no retest setup"

    # Long: break + retest at cap (band 0.05–0.30% below cap)
    if c > cap and d1h.close.iloc[-2] <= cap and trend4h in ("bull") and macd4 in ("up"):
        hi = cap * (1 - 0.0005)
        lo = cap * (1 - 0.0030)
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "LONG"
        note = "1H break + retest band below cap"

    # Short: break + retest at base (band 0.05–0.30% above base)
    if c < base and d1h.close.iloc[-2] >= base and trend4h in ("bear") and macd4 in ("down"):
        lo = base * (1 + 0.0005)
        hi = base * (1 + 0.0030)
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "SHORT"
        note = "1H break + retest band above base"

    return {
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
                "rsi": float(d1h.rsi.iloc[-1]),
                "macd": float(d1h.macd.iloc[-1]),
            },
            "zone": {
                "c": c,
                "donchian_high": float(cap),
                "donchian_low": float(base),
            },
        },
    }

# --- TICK (one scan + optional push) ---
def tick_once():
    ex = _exchange()
    results = []
    signals = []

    for sym in WATCHLIST:
        try:
            row = analyze_symbol(ex, sym)
            results.append(row)
            if row["signal"] in ("LONG", "SHORT"):
                signals.append(row)
        except Exception as e:
            results.append({"symbol": _mk(sym), "error": str(e)})

    # push signals to Worker -> Telegram
    if ALERT_WEBHOOK_URL and signals:
        payload = {
            "kind": "signals",
            "pass": PASS_KEY,
            "summary": f"{len(signals)} signal(s) from JIM engine",
            "signals": signals,
        }
        try:
            with httpx.Client(timeout=10.0) as cli:
                cli.post(ALERT_WEBHOOK_URL, json=payload)
        except Exception:
            pass

    return {
        "engine": "jim_v5_live",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": results,
        "note": "HTF→LTF + break+retest logic active",
    }
