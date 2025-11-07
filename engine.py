# engine.py — JIM v5.9.2 (global-ready) + latency logs
# - Backward compatible outputs for /tick and Worker
# - Toggle GLOBAL mode via env (no Supabase/Worker changes)

import os, json, time, pathlib
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

# -------- ENV --------
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").rstrip("/")
PASS_KEY          = os.getenv("PASS_KEY", "")

# Global mode controls
GLOBAL_MODE       = int(os.getenv("GLOBAL", "0"))  # 0=off, 1=on
EXCHANGES_ENV     = os.getenv("EXCHANGES", "binance,okx,bybit,kraken,coinbase")
EXCHANGES_LIST    = [e.strip().lower() for e in EXCHANGES_ENV.split(",") if e.strip()]

# Single-exchange fallback (used if GLOBAL=0, or if no global primary found)
PRICE_EXCHANGE    = os.getenv("PRICE_EXCHANGE", "mexc").lower()

WATCHLIST         = [s.strip() for s in os.getenv("WATCHLIST","BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if s.strip()]

# -------- helpers: exchange + latency --------
def _mk(symbol: str) -> str:
    # convert "BTCUSDT" -> "BTC/USDT"
    return symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"

def _get_exchange(name: str):
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True})

def timed_fetch_ohlcv(ex, symbol, timeframe="1h", limit=400):
    t0 = time.time()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    dt_ms = (time.time() - t0) * 1000.0
    print(f"[LATENCY] {ex.id.upper()} {symbol} {timeframe} x{limit} -> {dt_ms:.2f} ms")
    return data

def choose_primary_exchange():
    """
    GLOBAL_MODE=1:
      - Try EXCHANGES_LIST in order; return first that works for BTC/USDT.
    GLOBAL_MODE=0:
      - Use PRICE_EXCHANGE directly.
    """
    if not GLOBAL_MODE:
        try:
            ex = _get_exchange(PRICE_EXCHANGE)
            # quick symbol probe (BTC)
            _ = ex.load_markets()
            _ = ex.market(_mk("BTCUSDT"))
            print(f"[PRIMARY] single-exchange mode -> {ex.id}")
            return ex
        except Exception as e:
            raise RuntimeError(f"Primary exchange '{PRICE_EXCHANGE}' failed: {e}")

    # GLOBAL_MODE on: pick first healthy exchange
    for name in EXCHANGES_LIST:
        try:
            ex = _get_exchange(name)
            ex.load_markets()
            _ = ex.market(_mk("BTCUSDT"))
            print(f"[PRIMARY] global mode -> {ex.id}")
            return ex
        except Exception as _:
            continue
    # fallback: single-exchange
    ex = _get_exchange(PRICE_EXCHANGE)
    ex.load_markets()
    _ = ex.market(_mk("BTCUSDT"))
    print(f"[PRIMARY] fallback to {ex.id}")
    return ex

def extra_exchanges_for_volume(primary_id: str):
    """
    Build a list of extra exchanges (for aggregated volume) excluding primary.
    Only used when GLOBAL_MODE=1. Silent on failure.
    """
    if not GLOBAL_MODE:
        return []
    extras = []
    for name in EXCHANGES_LIST:
        if name == primary_id:
            continue
        try:
            ex = _get_exchange(name)
            ex.load_markets()
            extras.append(ex)
        except Exception:
            pass
    return extras

# -------- indicators --------
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
    # Use previous 4H candle high/low for anchors
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
    return _ohlcv(timed_fetch_ohlcv(ex, _mk(sym), timeframe=tf, limit=limit))

# -------- core analyze --------
def analyze_symbol(primary_ex, symbol, vol_extras):
    # price/structure from primary exchange
    d4h = fetch_tf(primary_ex, symbol, "4h", 300)
    d1h = fetch_tf(primary_ex, symbol, "1h", 400)

    # optional: aggregate global volume (last bar sum) without changing decisions
    vol_global = float(d1h["vol"].iloc[-1])
    if GLOBAL_MODE and vol_extras:
        for ex in vol_extras:
            try:
                df_ex = fetch_tf(ex, symbol, "1h", 2)  # last 2 bars only (fast)
                vol_global += float(df_ex["vol"].iloc[-1])
            except Exception as e:
                print(f"[VOLUME] skip {ex.id} {symbol}: {e}")

    # indicators
    d4h["ema21"] = ema(d4h.close, 21)
    d4h["ema50"] = ema(d4h.close, 50)
    d4h["macd"], d4h["macdsig"], _ = macd(d4h.close)
    trend4h = "bull" if d4h.ema21.iloc[-1] > d4h.ema50.iloc[-1] else "bear"
    macd4   = macd_slope(d4h["macd"])

    d1h["ema21"] = ema(d1h.close, 21)
    d1h["ema50"] = ema(d1h.close, 50)
    d1h["macd"], d1h["macdsig"], _ = macd(d1h.close)
    d1h["rsi"]   = rsi(d1h.close, 14)

    # ATR% (1h) for soft guard
    tr  = pd.concat([
        (d1h["high"] - d1h["low"]),
        (d1h["high"] - d1h["close"].shift()).abs(),
        (d1h["low"]  - d1h["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr1h_pct = float((atr.iloc[-1] / d1h["close"].iloc[-1]) * 100.0)

    # anchors
    cap, base = donchian(d4h, 55)
    c = float(d1h.close.iloc[-1])

    zone = "mid"
    if c >= cap: zone = "above_cap"
    elif c <= base: zone = "below_base"

    signal = "none"
    entry_band = None
    sl_max_pct = None
    note = "no retest setup"

    # Break + Retest logic (unchanged)
    # LONG: break above cap, then retest band 0.05–0.30% below cap
    if c > cap and d1h.close.iloc[-2] <= cap and trend4h == "bull" and macd4 == "up":
        hi = cap * (1 - 0.0005)
        lo = cap * (1 - 0.0030)
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "LONG"
        note = "1H break + retest band below cap"

    # SHORT: break below base, then retest band 0.05–0.30% above base
    if c < base and d1h.close.iloc[-2] >= base and trend4h == "bear" and macd4 == "down":
        lo = base * (1 + 0.0005)
        hi = base * (1 + 0.0030)
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "SHORT"
        note = "1H break + retest band above base"

    # momentum counters (for diagnostics only)
    mom_long = int(d1h.ema21.iloc[-1] > d1h.ema50.iloc[-1]) + int(d1h.rsi.iloc[-1] > 50)
    mom_short = int(d1h.ema21.iloc[-1] < d1h.ema50.iloc[-1]) + int(d1h.rsi.iloc[-1] < 50)

    # body break % (distance above/below anchor on current 1H close)
    body_break_pct = 0.0
    if c > cap:
        body_break_pct = (c - cap) / cap * 100.0
    elif c < base:
        body_break_pct = (base - c) / base * 100.0

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
                "mom_long": mom_long,
                "mom_short": mom_short,
                "body_break_pct": round(body_break_pct, 6),
                "macd4_slope": macd4,
                # diagnostics (global volume shown but NOT used in decisions)
                "vol_global_lastbar": vol_global if GLOBAL_MODE else float(d1h["vol"].iloc[-1]),
                "primary_exchange": primary_ex.id
            }
        },
    }
    return out

# -------- tick_once (one scan + optional push) --------
def tick_once():
    try:
        primary = choose_primary_exchange()
    except Exception as e:
        return {
            "engine": "jim_v5.9.2",
            "scanned": 0,
            "signals": [],
            "raw": [{"error": f"exchange init failed: {e}"}],
            "note": "init error"
        }

    volume_extras = extra_exchanges_for_volume(primary.id)

    results = []
    signals = []

    for sym in WATCHLIST:
        try:
            row = analyze_symbol(primary, sym, volume_extras)
            results.append(row)
            if row["signal"] in ("LONG", "SHORT"):
                signals.append({
                    "symbol": row["symbol"],
                    "side": row["signal"],
                    "entry": list(row["entry_zone"]) if row["entry_zone"] else None,
                    "sl": None,  # you can compute SL later or keep Worker rule-gate
                    "tp1": None, "tp2": None, "tp3": None,
                    "sl_max_pct": row["sl_max_pct"],
                    "note": row["note"]
                })
        except Exception as e:
            results.append({"symbol": _mk(sym), "error": str(e)})

    # Optional push to Worker (/signals)
    if ALERT_WEBHOOK_URL and signals:
        payload = {"pass": PASS_KEY, "summary": f"{len(signals)} signal(s) from JIM engine", "signals": signals}
        try:
            with httpx.Client(timeout=10.0) as cli:
                r = cli.post(ALERT_WEBHOOK_URL + "/signals", params={"t": PASS_KEY}, json=payload)
                print(f"[PUSH] worker /signals -> {r.status_code}")
        except Exception as e:
            print(f"[PUSH] worker error: {e}")

    return {
        "engine": "jim_v5.9.2",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": results,
        "note": "v5.9.2 structure+momentum engine active"
    }
