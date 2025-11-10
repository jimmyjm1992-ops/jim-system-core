# engine.py — JIM v5.9.3 (global-ready) + JSON-safe outputs
# - Adds Tier-2 momentum entries (continuation after clean break)
# - Tier labels in notes: "Tier: T1_SNIPER ..." or "Tier: T2_MOMENTUM ..."
# - Payload stable; adds helper fields for executors (sl_from_mid, sl_pct)
# - Quieter logs via LOG_VERBOSE; dual_ts(); tunable OHLC limits

import os, json, time, pathlib, math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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

# Logging + fetch sizes
LOG_VERBOSE     = int(os.getenv("LOG_VERBOSE", "0"))
OHLC_LIMIT_4H   = int(os.getenv("OHLC_LIMIT_4H", "300"))
OHLC_LIMIT_1H   = int(os.getenv("OHLC_LIMIT_1H", "400"))

# Global mode controls
GLOBAL_MODE    = int(os.getenv("GLOBAL", "0"))  # 0=off, 1=on
EXCHANGES_ENV  = os.getenv("EXCHANGES", "binance,okx,kraken,coinbase,mexc")
EXCHANGES_LIST = [e.strip().lower() for e in EXCHANGES_ENV.split(",") if e.strip()]

# Single-exchange fallback (used if GLOBAL=0, or if global primary not found)
PRICE_EXCHANGE = os.getenv("PRICE_EXCHANGE", "mexc").lower()

WATCHLIST = [s.strip() for s in os.getenv(
    "WATCHLIST",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,UNIUSDT,AAVEUSDT,ENAUSDT,TAOUSDT,PENDLEUSDT,VIRTUALUSDT,HBARUSDT,BCHUSDT,AVAXUSDT,LTCUSDT,DOTUSDT,SUIUSDT,NEARUSDT,OPUSDT,ARBUSDT,APTUSDT,ATOMUSDT,ALGOUSDT,INJUSDT,MATICUSDT,FTMUSDT,GALAUSDT,XLMUSDT,HYPEUSDT"
).split(",") if s.strip()]

# -------- util/logging --------
def log(msg: str):
    if LOG_VERBOSE:
        print(msg)

def fnum(x):
    """Return a JSON-safe float or None (filters NaN/Inf)."""
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None

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
    log(f"[LATENCY] {ex.id.upper()} {symbol} {timeframe} x{limit} -> {dt_ms:.2f} ms")
    return data

def choose_primary_exchange():
    """
    GLOBAL_MODE=1: try EXCHANGES_LIST in order; return first that supports BTC/USDT.
    GLOBAL_MODE=0: use PRICE_EXCHANGE directly.
    """
    if not GLOBAL_MODE:
        try:
            ex = _get_exchange(PRICE_EXCHANGE)
            ex.load_markets()
            _ = ex.market(_mk("BTCUSDT"))
            print(f"[PRIMARY] single-exchange mode -> {ex.id}")
            return ex
        except Exception as e:
            raise RuntimeError(f"Primary exchange '{PRICE_EXCHANGE}' failed: {e}")

    for name in EXCHANGES_LIST:
        try:
            ex = _get_exchange(name)
            ex.load_markets()
            _ = ex.market(_mk("BTCUSDT"))
            print(f"[PRIMARY] global mode -> {ex.id}")
            return ex
        except Exception:
            continue
    ex = _get_exchange(PRICE_EXCHANGE)
    ex.load_markets()
    _ = ex.market(_mk("BTCUSDT"))
    print(f"[PRIMARY] fallback to {ex.id}")
    return ex

def extra_exchanges_for_volume(primary_id: str):
    """Build list of extra exchanges (for aggregated volume) excluding primary."""
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

# -------- indicators (pandas) --------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    sigl = ema(line, sig)
    hist = line - sigl
    return line, sigl, hist

def rsi(close, n=14):
    d = close.diff()
    up = pd.Series(np.where(d > 0, d, 0.0), index=close.index).rolling(n).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0), index=close.index).rolling(n).mean().replace(0, np.nan)
    rs = up / dn
    out = 100 - (100 / (1 + rs))
    return out.bfill().ffill().fillna(50)

def donchian(df, n=55):
    # Use previous 4H candle high/low for anchors
    if len(df) < n + 2:
        return float("nan"), float("nan")
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
    d4h = fetch_tf(primary_ex, symbol, "4h", OHLC_LIMIT_4H)
    d1h = fetch_tf(primary_ex, symbol, "1h", OHLC_LIMIT_1H)

    # optional: aggregate global volume (last 1h bar sum) without changing decisions
    vol_global = float(d1h["vol"].iloc[-1])
    if GLOBAL_MODE and vol_extras:
        for ex in vol_extras:
            try:
                df_ex = fetch_tf(ex, symbol, "1h", 2)  # last 2 bars only (fast)
                vol_global += float(df_ex["vol"].iloc[-1])
            except Exception as e:
                log(f"[VOLUME] skip {ex.id} {symbol}: {e}")

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
    atr1h_pct = (atr.iloc[-1] / d1h["close"].iloc[-1]) * 100.0 if math.isfinite(atr.iloc[-1]) else float("nan")

    # anchors
    cap, base = donchian(d4h, 55)

    # If insufficient 4H history → return neutral, JSON-safe record
    if not np.isfinite(cap) or not np.isfinite(base):
        c = float(d1h.close.iloc[-1])
        return {
            "symbol": _mk(symbol),
            "price": fnum(c),
            "bias": "bearish" if d4h.close.iloc[-1] < d4h.close.iloc[-2] else "bullish",
            "zone": "mid",
            "signal": "none",
            "entry_zone": None,
            "sl_max_pct": None,
            "note": "insufficient 4H history for Donchian(55)",
            "why": {
                "bias": {
                    "c": fnum(c),
                    "ema21": fnum(d1h.close.ewm(span=21, adjust=False).mean().iloc[-1]),
                    "ema50": fnum(d1h.close.ewm(span=50, adjust=False).mean().iloc[-1]),
                    "rsi": fnum(rsi(d1h.close, 14).iloc[-1]),
                    "macd": fnum(macd(d1h.close)[0].iloc[-1]),
                },
                "zone": {"c": fnum(c), "donchian_high": None, "donchian_low": None},
                "metrics": {
                    "atr1h_pct": None,
                    "mom_long": None,
                    "mom_short": None,
                    "body_break_pct": None,
                    "macd4_slope": None,
                    "vol_global_lastbar": fnum(vol_global if GLOBAL_MODE else d1h["vol"].iloc[-1]),
                    "primary_exchange": primary_ex.id,
                },
            },
        }

    c = float(d1h.close.iloc[-1])

    zone = "mid"
    if c >= cap: zone = "above_cap"
    elif c <= base: zone = "below_base"

    signal = "none"
    entry_band = None
    sl_max_pct = None
    note = "no retest setup"

    # -------- Tier-1: Break + Retest (original) --------
    if c > cap and d1h.close.iloc[-2] <= cap and trend4h == "bull" and macd4 == "up":
        hi = cap * (1 - 0.0005)
        lo = cap * (1 - 0.0030)
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "LONG"
        note = "Tier: T1_SNIPER · 1H break + retest band below cap"

    if c < base and d1h.close.iloc[-2] >= base and trend4h == "bear" and macd4 == "down":
        lo = base * (1 + 0.0005)
        hi = base * (1 + 0.0030)
        entry_band = (lo, hi)
        sl_max_pct = 0.012
        signal = "SHORT"
        note = "Tier: T1_SNIPER · 1H break + retest band above base"

    # -------- Tier-2: Momentum continuation (conservative defaults) --------
    # Only if no T1 signal. Goal: catch continuation after clean break, with guardrails.
    if signal == "none":
        rsi_now = float(d1h["rsi"].iloc[-1])
        # safe windows
        atr_ok  = (atr1h_pct >= 0.5) and (atr1h_pct <= 2.0)
        body_ok = False
        body_break_pct = 0.0
        if c > cap:
            body_break_pct = (c - cap) / cap * 100.0
            body_ok = body_break_pct >= 0.25 and body_break_pct <= 1.20
        elif c < base:
            body_break_pct = (base - c) / base * 100.0
            body_ok = body_break_pct >= 0.25 and body_break_pct <= 1.20

        # Long continuation
        if c > cap and trend4h == "bull" and macd4 == "up" and 48 <= rsi_now <= 68 and atr_ok and body_ok:
            # micro pullback band around current price (tighter than T1)
            hi = c * (1 - 0.0008)
            lo = c * (1 - 0.0035)
            entry_band = (lo, hi)
            sl_max_pct = 0.012
            signal = "LONG"
            note = "Tier: T2_MOMENTUM · post-break continuation band (tight pullback)"

        # Short continuation
        if c < base and trend4h == "bear" and macd4 == "down" and 32 <= rsi_now <= 55 and atr_ok and body_ok:
            lo = c * (1 + 0.0008)
            hi = c * (1 + 0.0035)
            entry_band = (lo, hi)
            sl_max_pct = 0.012
            signal = "SHORT"
            note = "Tier: T2_MOMENTUM · post-break continuation band (tight pullback)"

    # momentum diagnostics
    mom_long  = int(d1h.ema21.iloc[-1] > d1h.ema50.iloc[-1]) + int(d1h.rsi.iloc[-1] > 50)
    mom_short = int(d1h.ema21.iloc[-1] < d1h.ema50.iloc[-1]) + int(d1h.rsi.iloc[-1] < 50)

    # body break %
    body_break_pct = 0.0
    if c > cap:
        body_break_pct = (c - cap) / cap * 100.0
    elif c < base:
        body_break_pct = (base - c) / base * 100.0

    out = {
        "symbol": _mk(symbol),
        "price": fnum(c),
        "bias": "bullish" if trend4h == "bull" else "bearish",
        "zone": zone,
        "signal": signal,
        "entry_zone": (fnum(entry_band[0]), fnum(entry_band[1])) if entry_band else None,
        "sl_max_pct": fnum(sl_max_pct),
        "note": note,
        "why": {
            "bias": {
                "c": fnum(c),
                "ema21": fnum(d1h.ema21.iloc[-1]),
                "ema50": fnum(d1h.ema50.iloc[-1]),
                "rsi": fnum(d1h.rsi.iloc[-1]),
                "macd": fnum(d1h.macd.iloc[-1]),
            },
            "zone": {
                "c": fnum(c),
                "donchian_high": fnum(cap),
                "donchian_low": fnum(base),
            },
            "metrics": {
                "atr1h_pct": fnum(atr1h_pct),
                "mom_long": int(mom_long),
                "mom_short": int(mom_short),
                "body_break_pct": fnum(body_break_pct),
                "macd4_slope": macd4,
                "vol_global_lastbar": fnum(vol_global if GLOBAL_MODE else d1h["vol"].iloc[-1]),
                "primary_exchange": primary_ex.id
            }
        },
    }
    return out

# -------- webhook helpers --------
def _signals_url():
    base = ALERT_WEBHOOK_URL.rstrip('/')
    return base if base.endswith('/signals') else f"{base}/signals"

def push_signals(signals: list):
    """Send signals array to Worker. Backward-compatible payload."""
    if not ALERT_WEBHOOK_URL or not signals:
        return
    payload = {
        "pass": PASS_KEY,
        "summary": f"{len(signals)} signal(s) from JIM engine",
        "signals": signals
    }
    try:
        with httpx.Client(timeout=10.0) as cli:
            url = _signals_url()
            # Keep old 't' param for workers relying on query validation
            r = cli.post(url, params={"t": PASS_KEY}, json=payload)
            ok = 200 <= r.status_code < 300
            print(f"[PUSH] {'OK' if ok else 'FAIL'} {url} -> {r.status_code} {(r.text or '')[:200]}")
    except Exception as e:
        print(f"[PUSH] EXC {_signals_url()} -> {e}")

# -------- tick_once (one scan + optional push) --------
def tick_once():
    try:
        primary = choose_primary_exchange()
    except Exception as e:
        return {
            "engine": "jim_v5.9.3",
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
                entry = list(row["entry_zone"]) if row["entry_zone"] else None
                entry_mid = None
                sl_from_mid = None
                if entry and entry[0] is not None and entry[1] is not None:
                    entry_mid = (entry[0] + entry[1]) / 2.0
                    if row["sl_max_pct"] is not None:
                        if row["signal"] == "LONG":
                            sl_from_mid = entry_mid * (1.0 - float(row["sl_max_pct"]))
                        else:
                            sl_from_mid = entry_mid * (1.0 + float(row["sl_max_pct"]))

                signals.append({
                    "symbol": row["symbol"],
                    "side": row["signal"],
                    "tf": "1h",
                    "price": row["price"],
                    "entry": entry,
                    "entry_mid": fnum(entry_mid),
                    # Provide both: mid-based SL (legacy) and % for client to compute from actual fill
                    "sl": fnum(sl_from_mid),
                    "sl_from_mid": fnum(sl_from_mid),
                    "sl_pct": fnum(row["sl_max_pct"]),  # e.g., 0.012 → client does fill*(1±0.012)
                    "tp1": None, "tp2": None, "tp3": None,
                    "sl_max_pct": row["sl_max_pct"],
                    "note": row["note"],
                    # Telemetry (safe to ignore by Worker)
                    "macd_slope": row["why"]["metrics"]["macd4_slope"],
                    "rsi": row["why"]["bias"]["rsi"],
                    "atr1h_pct": row["why"]["metrics"]["atr1h_pct"],
                    "body_break_pct": row["why"]["metrics"]["body_break_pct"]
                })
        except Exception as e:
            results.append({"symbol": _mk(sym), "error": str(e)})

    # Optional push to Worker
    if ALERT_WEBHOOK_URL and signals:
        push_signals(signals)

    return {
        "engine": "jim_v5.9.3",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": results,
        "note": "v5.9.3 T1+T2 engine active"
    }

# -------- helper for loop timestamps (used by engine_loop.py) --------
def dual_ts():
    utc = datetime.now(timezone.utc)
    kl  = utc.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
    return utc.strftime("%Y-%m-%d %H:%M:%S UTC") + " | " + kl.strftime("%Y-%m-%d %H:%M:%S KL")
