# ==========================================================
#  engine.py — JIM Framework v6.2 (Bernard-style alignment)
#
#  - 3 tiers: T1 Sniper (Break + Retest), T2 Momentum, T3 Pattern
#  - Bernard-style multi-touch horizontal S/R from swing points
#  - Optional BTC / BTC.D / USDT.D dominance filter (no Worker change)
#  - TP1/TP2/TP3 based on R-multiples from structure SL
#  - Safe for Koyeb free plan (1 exchange for prices + 1 for dominance)
#
#  JSON contract to Worker is UNCHANGED:
#    {symbol, side, tf, price, entry[], entry_mid, sl, tp1, tp2, tp3, note, ...}
# ==========================================================

import os, json, time, pathlib, math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ccxt
import httpx
from dotenv import load_dotenv

# ---------- FS ROOT ----------
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

load_dotenv()

# ---------- ENV ----------
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").rstrip("/")
PASS_KEY          = os.getenv("PASS_KEY", "")

LOG_VERBOSE     = int(os.getenv("LOG_VERBOSE", "0"))
OHLC_LIMIT_4H   = int(os.getenv("OHLC_LIMIT_4H", "260"))  # keep reasonable for CPU
OHLC_LIMIT_1H   = int(os.getenv("OHLC_LIMIT_1H", "260"))

PRICE_EXCHANGE  = os.getenv("PRICE_EXCHANGE", "mexc").lower()

# Optional dominance (safe even if missing)
DOM_EXCHANGE       = os.getenv("DOM_EXCHANGE", "binance").lower()
DOM_BTC_SYMBOL     = os.getenv("DOM_BTC_SYMBOL", "BTC/USDT")
DOM_BTCDOM_SYMBOL  = os.getenv("DOM_BTCDOM_SYMBOL", "")      # e.g. "BTCDOM/USDT"
DOM_USDTDOM_SYMBOL = os.getenv("DOM_USDTDOM_SYMBOL", "")     # e.g. "USDTDOM/USDT"

WATCHLIST = [
    s.strip() for s in os.getenv("WATCHLIST", "").split(",") if s.strip()
]

# ---------- UTILS ----------
def log(msg: str):
    if LOG_VERBOSE:
        print(msg)

def fnum(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None

def _mk(symbol: str) -> str:
    # "BTCUSDT" -> "BTC/USDT"
    if "/" in symbol:
        return symbol
    if len(symbol) > 4:
        return f"{symbol[:-4]}/{symbol[-4:]}"
    return symbol

def _get_exchange(name: str):
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True})


def timed_fetch_ohlcv(ex, symbol, timeframe="1h", limit=200):
    t0 = time.time()
    out = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    log(f"[LATENCY] {ex.id.upper()} {symbol} {timeframe} x{limit} -> {(time.time()-t0)*1000:.1f} ms")
    return out

def _ohlcv(klines):
    df = pd.DataFrame(klines, columns=["ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    return df

def fetch_tf(ex, sym, tf="1h", limit=200):
    return _ohlcv(timed_fetch_ohlcv(ex, _mk(sym), tf, limit))


# ---------- INDICATORS ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def rsi(close, n=14):
    diff = close.diff()
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(n).mean()
    roll_dn = pd.Series(dn, index=close.index).rolling(n).mean().replace(0, np.nan)
    rs = roll_up / roll_dn
    out = 100 - (100 / (1 + rs))
    return out.bfill().ffill().fillna(50.0)

def macd_slope(series):
    if len(series) < 2:
        return "flat"
    if series.iloc[-1] > series.iloc[-2]:
        return "up"
    if series.iloc[-1] < series.iloc[-2]:
        return "down"
    return "flat"

def atr_pct_1h(df1, period=14):
    tr = pd.concat(
        [
            df1["high"] - df1["low"],
            (df1["high"] - df1["close"].shift()).abs(),
            (df1["low"] - df1["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    c = df1["close"].iloc[-1]
    if not math.isfinite(atr) or not math.isfinite(c) or c == 0:
        return float("nan")
    return atr / c * 100.0


# ---------- SWING & S/R (Bernard-style) ----------
def detect_swing_levels(series, sensitivity=3):
    """
    Return list of (index_position, price) for swing highs & lows.
    Uses .iloc to avoid pandas FutureWarning.
    """
    n = len(series)
    out = []
    if n < sensitivity * 2 + 3:
        return out

    for pos in range(sensitivity, n - sensitivity):
        window = series.iloc[pos - sensitivity : pos + sensitivity + 1]
        val = series.iloc[pos]
        if val == window.max() or val == window.min():
            out.append((pos, float(val)))
    return out

def cluster_levels_from_swings(series, sensitivity=3, cluster_pct=0.003, min_touches=5):
    """
    Cluster swing points into horizontal levels by % distance.
    cluster_pct=0.003 => ~0.3% band.
    Returns sorted list of dicts: {"price": float, "points": [(pos, price), ...]}
    """
    swings = detect_swing_levels(series, sensitivity=sensitivity)
    clusters = []
    for pos, price in swings:
        assigned = False
        for cl in clusters:
            if abs(price - cl["price"]) / cl["price"] <= cluster_pct:
                cl["points"].append((pos, price))
                # update representative price: mean of points
                cl["price"] = float(np.mean([p for _, p in cl["points"]]))
                assigned = True
                break
        if not assigned:
            clusters.append({"price": price, "points": [(pos, price)]})

    dense = [cl for cl in clusters if len(cl["points"]) >= min_touches]
    dense.sort(key=lambda c: c["price"])
    return dense

def nearest_sr_levels(levels, ref_price):
    """
    Given sorted levels and current price, return (support, resistance)
    using Bernard's style: closest level below/above.
    """
    support = None
    resistance = None
    for cl in levels:
        p = cl["price"]
        if p <= ref_price:
            support = p
        if p > ref_price and resistance is None:
            resistance = p
    return support, resistance


# ---------- BTC / DOMINANCE REGIME ----------
def decide_risk_mode_from_dom():
    """
    Use BTC trend + BTC dominance + USDT dominance to decide risk mode.
    Falls back to 'auto' if symbols/exchange are unavailable.
    """
    try:
        ex = _get_exchange(DOM_EXCHANGE)
        ex.load_markets()
    except Exception as e:
        log(f"[DOM] cannot init {DOM_EXCHANGE}: {e}")
        return "auto"

    try:
        btc4 = fetch_tf(ex, DOM_BTC_SYMBOL, "4h", 120)
        btc1 = fetch_tf(ex, DOM_BTC_SYMBOL, "1h", 120)
    except Exception as e:
        log(f"[DOM] cannot fetch BTC: {e}")
        return "auto"

    # BTC trend by EMA stack on 4H
    btc4["ema21"] = ema(btc4["close"], 21)
    btc4["ema50"] = ema(btc4["close"], 50)
    btc_trend = "bull" if btc4["ema21"].iloc[-1] > btc4["ema50"].iloc[-1] else "bear"

    # Dominance directions (if tickers exist)
    def _dom_dir(symbol):
        if not symbol:
            return None
        try:
            d = fetch_tf(ex, symbol, "4h", 60)
            closes = d["close"]
            if closes.iloc[-1] > closes.iloc[-4]:
                return "up"
            if closes.iloc[-1] < closes.iloc[-4]:
                return "down"
            return "flat"
        except Exception as e:
            log(f"[DOM] fetch dom {symbol} failed: {e}")
            return None

    btc_dom_dir  = _dom_dir(DOM_BTCDOM_SYMBOL)
    usdt_dom_dir = _dom_dir(DOM_USDTDOM_SYMBOL)

    # If we don't have both dominance series, stay 'auto'
    if btc_dom_dir is None or usdt_dom_dir is None:
        return "auto"

    # Bernard-style intuition:
    # - BTC up, BTC.d down, USDT.d down  => strong risk-on (alts good)
    # - BTC down, BTC.d up, USDT.d up    => strong risk-off
    # - others => neutral (prefer with-trend only)
    if btc_trend == "bull" and btc_dom_dir == "down" and usdt_dom_dir == "down":
        return "risk_on"
    if btc_trend == "bear" and btc_dom_dir == "up" and usdt_dom_dir == "up":
        return "risk_off"

    # Mild neutrals:
    if btc_trend == "bull":
        return "neutral_long"
    if btc_trend == "bear":
        return "neutral_short"
    return "auto"


# ---------- TIER LOGIC ----------
def tier1_sniper(c1, support, resistance, d1, trend4, macd4, risk_mode):
    """
    Tier 1 — Sniper: Break + Retest at 5-touch level.
    - For LONG: price breaks resistance, retests, holds
    - For SHORT: price breaks support, retests, rejects
    """
    if support is None and resistance is None:
        return (None, None, None)

    close_prev = d1["close"].iloc[-2]

    # Only trade in allowed modes
    if risk_mode in ("risk_off",):
        return (None, None, None)

    # LONG sniper at resistance
    if resistance and trend4 == "bull" and macd4 == "up" and risk_mode in ("risk_on", "neutral_long", "auto"):
        broke = close_prev <= resistance and c1 > resistance
        if broke:
            hi = resistance * (1 - 0.0005)
            lo = resistance * (1 - 0.0030)
            return ("LONG", (lo, hi), "T1_SNIPER · Break + Retest at resistance")

    # SHORT sniper at support
    if support and trend4 == "bear" and macd4 == "down" and risk_mode in ("risk_on", "neutral_short", "auto"):
        broke = close_prev >= support and c1 < support
        if broke:
            lo = support * (1 + 0.0005)
            hi = support * (1 + 0.0030)
            return ("SHORT", (lo, hi), "T1_SNIPER · Break + Retest at support")

    return (None, None, None)


def tier2_momentum(c1, support, resistance, d1, trend4, macd4, atr_pct, risk_mode):
    """
    Tier 2 — Momentum: continuation after breakout.
    Conditions simplified but aligned to Bernard:
    - Only when ATR is 'reasonable'
    - Body size not too huge (avoid exhaustion) and not tiny (avoid chop)
    - RSI in acceptable band (no extreme overbought/oversold)
    """
    if not (0.4 <= atr_pct <= 2.5):
        return (None, None, None)

    rsi_val = float(d1["rsi"].iloc[-1])

    if risk_mode in ("risk_off",):
        return (None, None, None)

    last_open = d1["open"].iloc[-1]
    body_pct = abs(c1 - last_open) / last_open * 100.0 if last_open else 0.0

    # LONG continuation
    if trend4 == "bull" and macd4 == "up" and risk_mode in ("risk_on", "neutral_long", "auto"):
        if 0.25 <= body_pct <= 1.60 and 45 <= rsi_val <= 70:
            hi = c1 * (1 - 0.0008)
            lo = c1 * (1 - 0.0035)
            return ("LONG", (lo, hi), "T2_MOMENTUM · Continuation")

    # SHORT continuation
    if trend4 == "bear" and macd4 == "down" and risk_mode in ("risk_on", "neutral_short", "auto"):
        if 0.25 <= body_pct <= 1.60 and 30 <= rsi_val <= 55:
            lo = c1 * (1 + 0.0008)
            hi = c1 * (1 + 0.0035)
            return ("SHORT", (lo, hi), "T2_MOMENTUM · Continuation")

    return (None, None, None)


def tier3_pattern(c1, d1, levels_4h, risk_mode):
    """
    Tier 3 — Pattern: double bottom / top near 5-touch level.
    We stay in-with-trend / not risk-off.
    """
    if risk_mode == "risk_off":
        return (None, None, None)

    closes = d1["close"]
    n = len(closes)
    if n < 30:
        return (None, None, None)

    # Use last few swing points for double patterns
    swings_1h = detect_swing_levels(closes, sensitivity=3)
    if len(swings_1h) < 3:
        return (None, None, None)

    idx = [p[0] for p in swings_1h[-4:]]
    if len(idx) < 3:
        return (None, None, None)
    a, b, cpos = idx[-3], idx[-2], idx[-1]
    pa, pb = closes.iloc[a], closes.iloc[b]

    # Map to nearest 4H level to align with Bernard's S/R
    lvl_prices = [cl["price"] for cl in levels_4h] if levels_4h else []
    def nearest_level(price):
        if not lvl_prices:
            return None
        return min(lvl_prices, key=lambda lv: abs(lv - price))

    # Double Bottom pattern
    if abs(pa - pb) / pa < 0.006 and c1 > pb:
        base = nearest_level(pb) or pb
        lo = base * 0.995
        hi = c1 * 0.999
        return ("LONG", (lo, hi), "T3_PATTERN · Double Bottom near support")

    # Double Top pattern
    if abs(pa - pb) / pa < 0.006 and c1 < pb:
        base = nearest_level(pb) or pb
        hi = base * 1.005
        lo = c1 * 1.001
        return ("SHORT", (lo, hi), "T3_PATTERN · Double Top near resistance")

    return (None, None, None)


# ---------- MAIN ANALYZE PER SYMBOL ----------
def analyze_symbol(primary_ex, symbol, risk_mode):
    d4 = fetch_tf(primary_ex, symbol, "4h", OHLC_LIMIT_4H)
    d1 = fetch_tf(primary_ex, symbol, "1h", OHLC_LIMIT_1H)

    # 4H structure
    d4["ema21"] = ema(d4["close"], 21)
    d4["ema50"] = ema(d4["close"], 50)
    d4["macd"], _, _ = macd(d4["close"])
    trend4 = "bull" if d4["ema21"].iloc[-1] > d4["ema50"].iloc[-1] else "bear"
    macd4 = macd_slope(d4["macd"])

    # 1H structure
    d1["ema21"] = ema(d1["close"], 21)
    d1["ema50"] = ema(d1["close"], 50)
    d1["rsi"] = rsi(d1["close"])

    atr_pct = atr_pct_1h(d1)

    c1 = float(d1["close"].iloc[-1])
    c4 = float(d4["close"].iloc[-1])

    # 4H Bernard-style 5-touch levels
    levels_4h = cluster_levels_from_swings(d4["close"], sensitivity=3, cluster_pct=0.003, min_touches=5)
    support, resistance = nearest_sr_levels(levels_4h, c4)

    # ---- TIER 1 ----
    side, band, note = tier1_sniper(c1, support, resistance, d1, trend4, macd4, risk_mode)
    if side:
        return side, band, note, trend4, macd4, atr_pct

    # ---- TIER 2 ----
    side, band, note = tier2_momentum(c1, support, resistance, d1, trend4, macd4, atr_pct, risk_mode)
    if side:
        return side, band, note, trend4, macd4, atr_pct

    # ---- TIER 3 ----
    side, band, note = tier3_pattern(c1, d1, levels_4h, risk_mode)
    if side:
        return side, band, note, trend4, macd4, atr_pct

    return None, None, "no setup", trend4, macd4, atr_pct


# ---------- PUSH TO WORKER (UNCHANGED CONTRACT) ----------
def push_signals(signals):
    if not ALERT_WEBHOOK_URL or not signals:
        return
    url = ALERT_WEBHOOK_URL
    params = {"t": PASS_KEY}
    payload = {
        "summary": f"{len(signals)} signal(s) — JIM v6.2 Engine",
        "signals": signals,
    }
    try:
        with httpx.Client(timeout=10.0) as cli:
            r = cli.post(url, params=params, json=payload)
            print(f"[PUSH] {url} -> {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[PUSH ERROR] {e}")


# ---------- TICK ONCE ----------
def tick_once():
    # Decide global risk mode from dominance
    risk_mode = decide_risk_mode_from_dom()
    log(f"[RISK_MODE] {risk_mode}")

    try:
        ex = _get_exchange(PRICE_EXCHANGE)
        ex.load_markets()
    except Exception as e:
        return {"engine": "v6.2", "error": str(e)}

    signals = []
    raw = []

    for sym in WATCHLIST:
        sym_std = _mk(sym)
        row = {"symbol": sym_std}
        try:
            side, band, note, trend4, macd4, atr_pct = analyze_symbol(ex, sym, risk_mode)
            row.update(
                {
                    "note": note,
                    "trend4": trend4,
                    "macd4": macd4,
                    "atr1h_pct": atr_pct,
                }
            )

            if side and band:
                lo, hi = band
                entry_mid = (lo + hi) / 2.0
                c = entry_mid
                # Structure-based SL: ~1.5R away from band edge
                # (still tight, but not hard-coded %; Worker will still enforce max 1.26% if you keep that rule)
                if side == "LONG":
                    sl = lo * 0.992  # slightly below retest band
                else:
                    sl = hi * 1.008  # slightly above retest band

                risk_abs = abs(entry_mid - sl)
                sl_pct = risk_abs / entry_mid * 100.0 if entry_mid else None

                # TP based on R multiples
                if side == "LONG":
                    tp1 = entry_mid + risk_abs * 1.2
                    tp2 = entry_mid + risk_abs * 2.0
                    tp3 = entry_mid + risk_abs * 3.0
                else:
                    tp1 = entry_mid - risk_abs * 1.2
                    tp2 = entry_mid - risk_abs * 2.0
                    tp3 = entry_mid - risk_abs * 3.0

                signals.append(
                    {
                        "symbol": sym_std,
                        "side": side,
                        "tf": "1h",
                        "price": fnum(c),
                        "entry": [fnum(lo), fnum(hi)],
                        "entry_mid": fnum(entry_mid),
                        "sl": fnum(sl),
                        "tp1": fnum(tp1),
                        "tp2": fnum(tp2),
                        "tp3": fnum(tp3),
                        "note": note,
                        # telemetry for dashboard / Supabase (Worker unchanged)
                        "atr1h_pct": fnum(atr_pct),
                        "macd_slope": macd4,
                        "sl_max_pct": fnum(sl_pct),
                    }
                )

        except Exception as e:
            row["error"] = str(e)

        raw.append(row)

    if signals:
        push_signals(signals)

    return {
        "engine": "v6.2",
        "scanned": len(WATCHLIST),
        "risk_mode": risk_mode,
        "signals": signals,
        "raw": raw,
        "note": dual_ts(),
    }


# ---------- TIMESTAMP HELPER ----------
def dual_ts():
    utc = datetime.now(timezone.utc)
    kl = utc.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
    return f"{utc:%Y-%m-%d %H:%M:%S UTC} | {kl:%Y-%m-%d %H:%M:%S KL}"
