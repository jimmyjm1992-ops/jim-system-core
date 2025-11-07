# engine.py — JIM v5.9.2 (global data, structure + momentum)
# Compatible with your existing app.py, Worker (/signals), Supabase, Telegram

import os, json, math, pathlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt
import httpx

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# -------- ENV --------
ALERT_WEBHOOK_URL = (os.getenv("ALERT_WEBHOOK_URL", "") or "").rstrip("/")
PASS_KEY          = os.getenv("PASS_KEY", "")
# WATCHLIST accepts comma list without slashes (e.g., BTCUSDT) or with slashes (BTC/USDT)
WATCHLIST         = [s.strip() for s in os.getenv(
    "WATCHLIST",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
).split(",") if s.strip()]

# -------- UTILS --------
def _mk(sym: str) -> str:
    """Normalize 'BTCUSDT' -> 'BTC/USDT' """
    return sym if "/" in sym else f"{sym[:-4]}/{sym[-4:]}"

def _to_df(kl):
    df = pd.DataFrame(kl, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for c in ["open","high","low","close","vol"]:
        df[c] = df[c].astype(float)
    df = df.set_index("ts").sort_index()
    return df

def ema(s, n):
    return pd.Series(s).ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    sigl = ema(line, sig)
    hist = line - sigl
    return line, sigl, hist

def rsi(close, n=14):
    close = pd.Series(close)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean().replace(0, np.nan)
    rs = up / dn
    out = 100 - 100/(1+rs)
    return out.bfill().ffill().fillna(50)

def true_range(h, l, c_prev):
    return pd.concat([
        (h - l).abs(),
        (h - c_prev).abs(),
        (l - c_prev).abs()
    ], axis=1).max(axis=1)

def atr(df, n=14):
    c_prev = df["close"].shift(1)
    tr = true_range(df["high"], df["low"], c_prev)
    return tr.rolling(n).mean()

def donchian_levels(df_4h, lookback=55):
    # Use previous 4H candle cap/base to avoid current-candle peek
    cap  = df_4h["high"].rolling(lookback).max().iloc[-2]
    base = df_4h["low"].rolling(lookback).min().iloc[-2]
    return float(cap), float(base)

def macd_slope(series):
    if len(series) < 2:
        return "flat"
    return "up" if series.iloc[-1] > series.iloc[-2] else ("down" if series.iloc[-1] < series.iloc[-2] else "flat")

# -------- EXCHANGES (global median) --------
# We instantiate 4 exchanges; fetch OHLCV concurrently (serial here but fast enough with enableRateLimit).
# Any failing or region-blocked source is skipped; we aggregate the rest.

_EX_NAMES = ("binance", "okx", "bybit", "mexc")
def _ex_instances():
    exes = {}
    for name in _EX_NAMES:
        try:
            cls = getattr(ccxt, name)
            exes[name] = cls({"enableRateLimit": True})
        except Exception:
            pass
    return exes

EXES = _ex_instances()

def _fetch_ohlcv_one(ex, symbol, tf, limit):
    try:
        return _to_df(ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit))
    except Exception:
        return None

def _align_and_median(dfs: list):
    """Align by timestamp and take median for open/high/low/close; sum for volume."""
    if not dfs:
        return None
    # Keep only non-None
    dfs = [d for d in dfs if d is not None and len(d) > 0]
    if not dfs:
        return None
    # Outer join on index, with suffixes; then take row-wise median for O/H/L/C and sum for vol.
    # Easier approach: reindex to union, then per column take median across frames.
    idx = None
    for d in dfs:
        idx = d.index if idx is None else idx.union(d.index)
    cols = ["open","high","low","close","vol"]
    stacked = {c: [] for c in cols}
    for d in dfs:
        dd = d.reindex(idx)
        for c in cols:
            stacked[c].append(dd[c])
    out = pd.DataFrame(index=idx)
    for c in ["open","high","low","close"]:
        out[c] = pd.concat(stacked[c], axis=1).median(axis=1, skipna=True)
    out["vol"] = pd.concat(stacked["vol"], axis=1).sum(axis=1, skipna=True)
    out = out.dropna().sort_index()
    return out

def fetch_global(symbol, tf="1h", limit=400):
    """Fetch & aggregate OHLCV from multiple spot exchanges for a global view."""
    dfs = []
    for name, ex in EXES.items():
        d = _fetch_ohlcv_one(ex, symbol, tf, limit)
        if d is not None and len(d) > 0:
            dfs.append(d)
    return _align_and_median(dfs)

# -------- ANALYZE --------
def analyze_symbol(symbol):
    sym = _mk(symbol)

    d1h = fetch_global(sym, "1h", 400)
    d4h = fetch_global(sym, "4h", 300)

    if d1h is None or d4h is None or len(d1h) < 60 or len(d4h) < 20:
        return {"symbol": sym, "error": "insufficient global data"}

    # Indicators
    d4h["ema21"] = ema(d4h.close, 21)
    d4h["ema50"] = ema(d4h.close, 50)
    d4h["macd"], d4h["macdsig"], _ = macd(d4h.close)

    trend4h = "bull" if d4h.ema21.iloc[-1] > d4h.ema50.iloc[-1] else "bear"
    macd4_s = macd_slope(d4h["macd"])

    d1h["ema21"] = ema(d1h.close, 21)
    d1h["ema50"] = ema(d1h.close, 50)
    d1h["macd"], d1h["macdsig"], _ = macd(d1h.close)
    d1h["rsi"] = rsi(d1h.close, 14)

    # ATR% (1h)
    d1h["atr"] = atr(pd.DataFrame({
        "high": d1h["high"],
        "low": d1h["low"],
        "close": d1h["close"]
    }), 14)
    atr1h_pct = float(d1h["atr"].iloc[-1] / d1h["close"].iloc[-1] * 100.0)

    # Donchian (55) on 4h
    cap, base = donchian_levels(d4h, 55)

    c = float(d1h.close.iloc[-1])
    prev_close = float(d1h.close.iloc[-2])

    zone = "mid"
    if c >= cap:
        zone = "above_cap"
    elif c <= base:
        zone = "below_base"

    # Momentum counters (0..2)
    mom_long = 0
    mom_short = 0
    if d1h.rsi.iloc[-1] > 50: mom_long += 1
    if d1h.rsi.iloc[-1] < 50: mom_short += 1
    if macd4_s == "up": mom_long += 1
    if macd4_s == "down": mom_short += 1

    # Body-break % (last 1h body close beyond cap/base)
    body_break_pct = 0.0
    if prev_close <= cap and c > cap:
        body_break_pct = (c - cap) / cap * 100.0
    elif prev_close >= base and c < base:
        body_break_pct = (base - c) / base * 100.0

    # Signal logic (unchanged rules: break + retest)
    signal = "none"
    entry_band = None
    sl_max_pct = None
    note = "no retest setup"

    # LONG: break above cap + retest band below cap
    if c > cap and prev_close <= cap and trend4h == "bull" and macd4_s == "up":
        hi = cap * (1 - 0.0005)  # -0.05%
        lo = cap * (1 - 0.0030)  # -0.30%
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "LONG"
        note = "1H break + retest band below cap"

    # SHORT: break below base + retest band above base
    if c < base and prev_close >= base and trend4h == "bear" and macd4_s == "down":
        lo = base * (1 + 0.0005)  # +0.05%
        hi = base * (1 + 0.0030)  # +0.30%
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "SHORT"
        note = "1H break + retest band above base"

    row = {
        "symbol": sym,
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
            "metrics": {
                "atr1h_pct": atr1h_pct,
                "mom_long": int(mom_long),
                "mom_short": int(mom_short),
                "body_break_pct": float(body_break_pct),
                "macd4_slope": macd4_s,
            }
        }
    }
    return row

# -------- TICK --------
def _post_signals(signals):
    if not ALERT_WEBHOOK_URL or not signals:
        return
    # Accept both base path or base + ?t=PASS in env
    url = ALERT_WEBHOOK_URL
    if "?" not in url:
        # assume Worker expects /signals?t=PASS
        if not url.endswith("/signals"):
            url = url.rstrip("/") + "/signals"
        url = f"{url}?t={PASS_KEY}"

    payload = {
        "kind": "signals",
        "pass": PASS_KEY,
        "summary": f"{len(signals)} signal(s) from JIM engine",
        "signals": signals,
    }
    try:
        with httpx.Client(timeout=12.0) as cli:
            cli.post(url, json=payload)
    except Exception:
        # do not raise; engine must continue
        pass

def tick_once():
    results = []
    signals = []
    for sym in WATCHLIST:
        try:
            row = analyze_symbol(sym)
            results.append(row)
            if row.get("signal") in ("LONG", "SHORT"):
                # flatten structure for worker insert schema
                s = {
                    "symbol": row["symbol"],
                    "side": row["signal"],
                    "price": row["price"],
                    "entry": row["entry_zone"],
                    "sl_max_pct": row.get("sl_max_pct"),
                    "note": row.get("note"),
                    # optional enrich (worker will accept)
                    "rsi": row["why"]["bias"]["rsi"],
                    "macd_slope": row["why"]["metrics"]["macd4_slope"],
                    "tp1": None, "tp2": None, "tp3": None,  # left blank by engine
                    "sl": None,                               # SL placement is tactical
                    "timeframe": "1h"
                }
                signals.append(s)
        except Exception as e:
            results.append({"symbol": _mk(sym), "error": str(e)})

    # push out
    _post_signals(signals)

    return {
        "engine": "jim_v5.9.2",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": results,
        "note": "v5.9.2 structure+momentum engine active",
    }
