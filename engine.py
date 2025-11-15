# ==========================================================
#  engine.py — Bernard System v2.3 (Pure Bernard Logic)
#
#  - 3 Bernard families:
#       B1: Breakout + Retest at strong 4H S/R
#       B2: Trend Continuation (EMA pullback / mini-flag)
#       B3: Major S/R Bounce + Pattern (double top/bottom + SFP-style)
#
#  - BTC regime filter (risk_on / risk_off / neutral)
#  - Optional dominance bias (BTCDOM + USDTDOM via env)
#  - Structure-based SL, price risk capped at 5%
#    (so with 10% margin & 10x lev, account loss ≤ 5%)
#  - TP = 1R / 2R / 3R
#  - Safe for Koyeb free plan (4H + 1H only, hourly run)
#  - Fully compatible with Worker & Supabase contracts
# ==========================================================

import os, time, pathlib, math
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
OHLC_LIMIT_4H   = int(os.getenv("OHLC_LIMIT_4H", "300"))
OHLC_LIMIT_1H   = int(os.getenv("OHLC_LIMIT_1H", "400"))

PRICE_EXCHANGE  = os.getenv("PRICE_EXCHANGE", "mexc").lower()

# Optional dominance envs (safe defaults)
DOM_EXCHANGE        = os.getenv("DOM_EXCHANGE", PRICE_EXCHANGE).lower()
DOM_BTCDOM_SYMBOL   = os.getenv("DOM_BTCDOM_SYMBOL", "").strip()
DOM_USDTDOM_SYMBOL  = os.getenv("DOM_USDTDOM_SYMBOL", "").strip()

WATCHLIST = [
    s.strip()
    for s in os.getenv(
        "WATCHLIST",
        # default 30; you already extended this to 100 in Koyeb env
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,UNIUSDT,AAVEUSDT,"
        "ENAUSDT,TAOUSDT,PENDLEUSDT,VIRTUALUSDT,HBARUSDT,BCHUSDT,AVAXUSDT,"
        "LTCUSDT,DOTUSDT,SUIUSDT,NEARUSDT,OPUSDT,ARBUSDT,APTUSDT,ATOMUSDT,"
        "ALGOUSDT,INJUSDT,MATICUSDT,FTMUSDT,GALAUSDT,XLMUSDT,HYPEUSDT"
    ).split(",")
    if s.strip()
]

# Max allowed price risk (in %) between entry_mid and SL
# With 10% capital, 10x leverage, this equals max 5% account loss.
MAX_PRICE_RISK_PCT = 5.0

# Minimum distance (in ATR multiples) from B2 entry price to nearest 4H S/R
# to avoid shorting directly into strong support or longing into strong resistance.
MIN_B2_SR_ATR_MULT = 1.5


# ---------- Utils ----------
def log(msg: str):
    if LOG_VERBOSE:
        print(msg)


def fnum(x):
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def pct_distance(a: float, b: float) -> float:
    """% distance between two prices, relative to mid-price."""
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return 1e9
    mid = (a + b) / 2.0
    if mid <= 0:
        return 1e9
    return abs(a - b) / mid * 100.0


def _mk(symbol: str) -> str:
    """Convert BTCUSDT -> BTC/USDT for ccxt."""
    return symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"


def _get_exchange(name: str):
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True})


# ---------- OHLC helpers ----------
def timed_fetch_ohlcv(ex, symbol: str, timeframe: str = "1h", limit: int = 400):
    t0 = time.time()
    out = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    log(f"[LATENCY] {ex.id.upper()} {symbol} {timeframe} x{limit} -> {(time.time()-t0)*1000:.2f} ms")
    return out


def _ohlcv(kl):
    df = pd.DataFrame(kl, columns=["ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    return df


def fetch_tf(ex, sym: str, tf: str = "1h", limit: int = 400):
    return _ohlcv(timed_fetch_ohlcv(ex, _mk(sym), tf, limit))


# ---------- Indicators (Bernard-style) ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


# Bernard often uses a "faster" MACD; we use (5,20,9)
def macd(close: pd.Series, fast: int = 5, slow: int = 20, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = np.where(d > 0, d, 0)
    dn = np.where(d < 0, -d, 0)
    roll_up = pd.Series(up, index=close.index).rolling(n).mean()
    roll_dn = pd.Series(dn, index=close.index).rolling(n).mean().replace(0, np.nan)
    rs = roll_up / roll_dn
    out = 100 - (100 / (1 + rs))
    return out.bfill().ffill().fillna(50)


def atr(df: pd.DataFrame, n: int = 14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def macd_slope(series: pd.Series) -> str:
    if len(series) < 2:
        return "flat"
    if series.iloc[-1] > series.iloc[-2]:
        return "up"
    if series.iloc[-1] < series.iloc[-2]:
        return "down"
    return "flat"

# ---------- Swings & S/R ----------
def swing_structure_1h(close: pd.Series, look: int = 2):
    """
    Very small helper to read 1H swing structure for B2:
    - looks at last 2 swing highs and last 2 swing lows
    - returns 'up' / 'down' / 'flat' for highs and lows
    """
    swings = detect_swings(close, look=look)
    high_idx = [i for (i, p, t) in swings if t == "high"]
    low_idx  = [i for (i, p, t) in swings if t == "low"]

    def trend_from_idx(idxs):
        if len(idxs) < 2:
            return "flat"
        a, b = idxs[-2], idxs[-1]
        pa, pb = float(close.iloc[a]), float(close.iloc[b])
        if pb > pa * 1.001:
            return "up"
        if pb < pa * 0.999:
            return "down"
        return "flat"

    return {
        "high_trend": trend_from_idx(high_idx),
        "low_trend": trend_from_idx(low_idx),
    }

def cluster_levels(swings, level_type: str, tolerance_pct: float = 0.4, min_touches: int = 4):
    """
    Cluster swing highs/lows into S/R levels.
    tolerance_pct: max distance within cluster (% of price).
    Returns list of (price, touches).
    """
    vals = [p for (_, p, t) in swings if t == level_type]
    if not vals:
        return []

    vals = sorted(vals)
    clusters = [[vals[0], 1]]  # [avg, count]

    for v in vals[1:]:
        avg, cnt = clusters[-1]
        if abs(v - avg) / avg * 100 <= tolerance_pct:
            new_avg = (avg * cnt + v) / (cnt + 1)
            clusters[-1] = [new_avg, cnt + 1]
        else:
            clusters.append([v, 1])

    out = [(avg, cnt) for (avg, cnt) in clusters if cnt >= min_touches]
    return out


def nearest_level(levels, price: float, direction: str):
    """
    direction: 'above' for resistance, 'below' for support.
    levels: list of (price, touches)
    """
    if not levels:
        return None, 0
    if direction == "above":
        candidates = [(lv, n) for (lv, n) in levels if lv >= price]
        if not candidates:
            return None, 0
        lv, n = min(candidates, key=lambda x: x[0])
        return lv, n
    else:
        candidates = [(lv, n) for (lv, n) in levels if lv <= price]
        if not candidates:
            return None, 0
        lv, n = max(candidates, key=lambda x: x[0])
        return lv, n


# ---------- Trend classification ----------
def classify_trend(close: pd.Series, e10: pd.Series, e21: pd.Series, e50: pd.Series, s200: pd.Series):
    """
    Bernard EMA stack trend:
    - up: close > 10 > 21 > 50 > 200
    - down: close < 10 < 21 < 50 < 200
    Else: chop
    """
    if len(close) < 200:
        return "chop"

    c = close.iloc[-1]
    v10 = e10.iloc[-1]
    v21 = e21.iloc[-1]
    v50 = e50.iloc[-1]
    v200 = s200.iloc[-1]

    if c > v10 > v21 > v50 > v200:
        return "up"
    if c < v10 < v21 < v50 < v200:
        return "down"
    return "chop"


# ---------- BTC Regime + Dominance ----------
def analyze_btc_regime(ex_price, ex_dom=None):
    """
    BTC regime filter:
      - risk_on: BTC uptrend, MACD up, RSI>50
      - risk_off: BTC downtrend, MACD down, RSI<50
      - neutral: everything else

    Dominance (optional):
      - if env dominance symbols configured, derive a simple bias.
    """
    regime = "neutral"
    dom_bias = "neutral"

    try:
        df4 = fetch_tf(ex_price, "BTCUSDT", "4h", OHLC_LIMIT_4H)
        df1 = fetch_tf(ex_price, "BTCUSDT", "1h", OHLC_LIMIT_1H)

        df4["ema10"] = ema(df4.close, 10)
        df4["ema21"] = ema(df4.close, 21)
        df4["ema50"] = ema(df4.close, 50)
        df4["sma200"] = sma(df4.close, 200)
        df4["macd"], _, _ = macd(df4.close)

        df1["ema10"] = ema(df1.close, 10)
        df1["ema21"] = ema(df1.close, 21)
        df1["ema50"] = ema(df1.close, 50)
        df1["sma200"] = sma(df1.close, 200)
        df1["macd"], _, _ = macd(df1.close)
        df1["rsi"] = rsi(df1.close)

        trend4 = classify_trend(df4.close, df4.ema10, df4.ema21, df4.ema50, df4.sma200)
        trend1 = classify_trend(df1.close, df1.ema10, df1.ema21, df1.ema50, df1.sma200)
        macd1_slope = macd_slope(df1["macd"])
        rsi1 = float(df1["rsi"].iloc[-1])

        if trend4 == "up" and trend1 in ("up", "chop") and macd1_slope == "up" and rsi1 > 50:
            regime = "risk_on"
        elif trend4 == "down" and trend1 in ("down", "chop") and macd1_slope == "down" and rsi1 < 50:
            regime = "risk_off"
        else:
            regime = "neutral"
    except Exception as e:
        log(f"[BTC_REGIME] error: {e}")
        regime = "neutral"

    # optional dominance
    if ex_dom is not None and DOM_BTCDOM_SYMBOL and DOM_USDTDOM_SYMBOL:
        try:
            btcd = fetch_tf(ex_dom, DOM_BTCDOM_SYMBOL, "4h", OHLC_LIMIT_4H)
            usdt = fetch_tf(ex_dom, DOM_USDTDOM_SYMBOL, "4h", OHLC_LIMIT_4H)
            btcd["ema20"] = ema(btcd.close, 20)
            usdt["ema20"] = ema(usdt.close, 20)

            btcd_trend = "up" if btcd.close.iloc[-1] > btcd.ema20.iloc[-1] else "down"
            usdt_trend = "up" if usdt.close.iloc[-1] > usdt.ema20.iloc[-1] else "down"

            if btcd_trend == "up" and usdt_trend == "down":
                dom_bias = "btc_pump_usdt_dump"
            elif btcd_trend == "down" and usdt_trend == "up":
                dom_bias = "btc_dump_usdt_pump"
            else:
                dom_bias = "neutral"
        except Exception as e:
            log(f"[DOM] error: {e}")
            dom_bias = "neutral"

    return regime, dom_bias

# ---------- Candlestick helpers (v2.3) ----------
def candle_stats(row):
    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        rng = 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_pct = body / rng * 100.0
    return o, h, l, c, body, rng, upper_wick, lower_wick, body_pct


def is_bull_engulf(prev, row):
    po, ph, pl, pc, _, _, _, _, _ = candle_stats(prev)
    o, h, l, c, _, _, _, _, _ = candle_stats(row)
    # previous red, current green and engulfs body
    return (pc < po) and (c > o) and (c >= po) and (o <= pc)


def is_bear_engulf(prev, row):
    po, ph, pl, pc, _, _, _, _, _ = candle_stats(prev)
    o, h, l, c, _, _, _, _, _ = candle_stats(row)
    # previous green, current red and engulfs body
    return (pc > po) and (c < o) and (c <= po) and (o >= pc)


def is_bull_pin(row):
    o, h, l, c, body, rng, uw, lw, body_pct = candle_stats(row)
    # long lower wick, small body near top
    return lw > 2 * body and body_pct > 15 and min(o, c) > l + rng * 0.4


def is_bear_pin(row):
    o, h, l, c, body, rng, uw, lw, body_pct = candle_stats(row)
    # long upper wick, small body near bottom
    return uw > 2 * body and body_pct > 15 and max(o, c) < h - rng * 0.4


def is_inside_bar(prev, row):
    return row["high"] <= prev["high"] and row["low"] >= prev["low"]


def is_strong_break_candle(prev, row, level: float, direction: str, min_body_atr_pct: float, atr_val: float):
    """
    Strong breakout candle:
    - body size >= min_body_atr_pct of ATR
    - close clearly beyond level
    - wick not fighting breakout direction too much
    """
    o, h, l, c, body, rng, uw, lw, _ = candle_stats(row)
    if atr_val is None or atr_val <= 0:
        return False
    body_atr_pct = body / atr_val * 100.0
    if body_atr_pct < min_body_atr_pct:
        return False

    if direction == "up":
        if c <= level:
            return False
        # avoid huge upper wick against direction
        if uw / rng > 0.5:
            return False
    else:
        if c >= level:
            return False
        if lw / rng > 0.5:
            return False
    return True


def bullish_retest_candle(row, level: float, tolerance_pct: float = 0.4):
    """
    Retest candle for LONG (B1):
    - lower wick tags level
    - close > open
    - close above level
    """
    o, h, l, c, _, rng, uw, lw, _ = candle_stats(row)
    if l <= level * (1 + tolerance_pct / 100) and h >= level * (1 - tolerance_pct / 100):
        if c > o and c > level:
            return True
    return False


def bearish_retest_candle(row, level: float, tolerance_pct: float = 0.4):
    """
    Retest candle for SHORT (B1):
    - upper wick tags level
    - close < open
    - close below level
    """
    o, h, l, c, _, rng, uw, lw, _ = candle_stats(row)
    if h >= level * (1 - tolerance_pct / 100) and l <= level * (1 + tolerance_pct / 100):
        if c < o and c < level:
            return True
    return False


def is_rejection_candle(row, at_support: bool, level: float, tolerance_pct: float = 0.5):
    """
    Rejection candle for B3 (bounce):
    - Long wick into level
    - Strong body away from level
    """
    o, h, l, c, body, rng, uw, lw, body_pct = candle_stats(row)
    if rng <= 0:
        return False

    if at_support:
        lower_wick = lw
        if (
            l <= level * (1 + tolerance_pct / 100)
            and lower_wick / rng > 0.4
            and c > o
            and body_pct > 20
        ):
            return True
    else:
        upper_wick = uw
        if (
            h >= level * (1 - tolerance_pct / 100)
            and upper_wick / rng > 0.4
            and c < o
            and body_pct > 20
        ):
            return True

    return False


# ---------- Simple SFP / liquidity sweep helpers ----------
def detect_sfp_high(d1: pd.DataFrame, swings, lookback: int = 20):
    """
    Detect simple swing-failure pattern at highs:
    - current high takes previous swing high
    - close back inside
    """
    if len(d1) < 5 or not swings:
        return False
    last_idx = len(d1) - 1
    recent_swings = [i for (i, p, t) in swings if t == "high" and last_idx - lookback <= i < last_idx]
    if not recent_swings:
        return False
    last = d1.iloc[-1]
    h = last["high"]
    c = last["close"]
    prev_idx = recent_swings[-1]
    prev_high = float(d1["high"].iloc[prev_idx])
    # sweep previous high but close back under
    return h > prev_high * 1.0005 and c < prev_high


def detect_sfp_low(d1: pd.DataFrame, swings, lookback: int = 20):
    """
    Detect simple swing-failure pattern at lows:
    - current low takes previous swing low
    - close back inside
    """
    if len(d1) < 5 or not swings:
        return False
    last_idx = len(d1) - 1
    recent_swings = [i for (i, p, t) in swings if t == "low" and last_idx - lookback <= i < last_idx]
    if not recent_swings:
        return False
    last = d1.iloc[-1]
    l = last["low"]
    c = last["close"]
    prev_idx = recent_swings[-1]
    prev_low = float(d1["low"].iloc[prev_idx])
    return l < prev_low * 0.9995 and c > prev_low

# ---------- B1: Breakout + Retest ----------
def b1_break_retest(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    """
    B1: Breakout / Breakdown with Retest at strong 4H S/R.
    Now with:
      - strong breakout candle check
      - MACD/RSI filter
      - retest candle pattern filter (engulf / pin / rejection)
    """
    highs = cluster_levels(swings4, "high", tolerance_pct=0.4, min_touches=5)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=0.4, min_touches=5)

    c1 = float(d1["close"].iloc[-1])
    prev = d1.iloc[-2]
    last = d1.iloc[-1]

    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    rsi1 = float(d1["rsi"].iloc[-1])
    atr_val = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None

    # ---- LONG: breakout above resistance, retest, bullish confirmation
    if regime != "risk_off" and trend4 == "up" and dom_bias != "btc_pump_usdt_dump":
        if res_level:
            prev_close = float(prev["close"])
            # check strong break candle (prev) across level
            if prev_close > res_level * 1.001 and float(prev["open"]) <= res_level:
                if not is_strong_break_candle(
                    d1.iloc[-3],
                    prev,
                    res_level,
                    "up",
                    min_body_atr_pct=40,
                    atr_val=atr_val,
                ):
                    return None
                # now last candle is retest candle
                if not (
                    bullish_retest_candle(last, res_level)
                    or is_bull_engulf(prev, last)
                    or is_bull_pin(last)
                ):
                    return None
                if not (45 <= rsi1 <= 70 and macd1_dir in ("up", "flat")):
                    return None

                entry_low = max(res_level, last["open"])
                entry_high = last["close"] * 1.0015
                entry_mid = (entry_low + entry_high) / 2.0

                sl_price = min(last["low"], res_level * 0.997)
                price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100

                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    return {
                        "side": "LONG",
                        "entry_low": entry_low,
                        "entry_high": entry_high,
                        "sl": sl_price,
                        "note": f"B1_BREAK_RETEST · {res_touch}x resistance @ {res_level:.4f}",
                    }

    # ---- SHORT: breakdown below support, retest, bearish confirmation
    if trend4 == "down":
        if sup_level:
            prev_close = float(prev["close"])
            if prev_close < sup_level * 0.999 and float(prev["open"]) >= sup_level:
                if not is_strong_break_candle(
                    d1.iloc[-3],
                    prev,
                    sup_level,
                    "down",
                    min_body_atr_pct=40,
                    atr_val=atr_val,
                ):
                    return None
                if not (
                    bearish_retest_candle(last, sup_level)
                    or is_bear_engulf(prev, last)
                    or is_bear_pin(last)
                ):
                    return None
                if not (30 <= rsi1 <= 60 and macd1_dir in ("down", "flat")):
                    return None

                entry_high = min(sup_level, last["open"])
                entry_low = last["close"] * 0.9985
                entry_mid = (entry_low + entry_high) / 2.0

                sl_price = max(last["high"], sup_level * 1.003)
                price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100

                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    return {
                        "side": "SHORT",
                        "entry_low": entry_low,
                        "entry_high": entry_high,
                        "sl": sl_price,
                        "note": f"B1_BREAK_RETEST · {sup_touch}x support @ {sup_level:.4f}",
                    }

    return None

# ---------- B2: Trend Continuation (EMA pullback / mini-flag) ----------
def b2_trend_continuation(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    """
    B2 v2.1: Trend continuation (Bernard-style) using EMA pullback / mini-flag.

    Extra filters vs v2.0:
    - 4H S/R distance filter (avoid shorting into strong support / longing into strong resistance)
    - 1H swing structure (must be LH/LL for shorts, HH/HL for longs)
    - Stricter MACD/RSI confirmation for trend continuation
    - Optional volume behaviour check (flag consolidation has lighter volume than impulse,
      trigger candle volume not dead)
    """
    # 1) 4H trend must be clear up or down
    if trend4 not in ("up", "down"):
        return None

    # 2) Global regime guard (still allow shorts in risk_off)
    if regime == "risk_off" and trend4 == "up":
        return None

    last = d1.iloc[-1]
    prev = d1.iloc[-2]

    c = float(last["close"])
    atr1 = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None
    if not atr1 or atr1 <= 0:
        return None

    # 3) Flag-style consolidation: last 4 candles before trigger
    recent = d1.iloc[-5:-1]  # last 4 before current
    if len(recent) < 4:
        return None

    ranges = (recent["high"] - recent["low"]).abs()
    bodies = (recent["close"] - recent["open"]).abs()
    avg_range = float(ranges.mean())
    avg_body = float(bodies.mean())

    # Want relatively small-body consolidation vs total range
    if avg_range <= 0 or avg_body / avg_range > 0.7:
        return None

    # 4) MACD + RSI (stricter continuation rules)
    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    macd_val = float(macd_line.iloc[-1])
    rsi1 = float(d1["rsi"].iloc[-1])

    ema10_1h = float(d1["ema10"].iloc[-1])
    ema21_1h = float(d1["ema21"].iloc[-1])
    ema50_1h = float(d1["ema50"].iloc[-1])

    # 5) 1H swing structure: we only want continuation, not hidden reversal
    struct1h = swing_structure_1h(d1["close"], look=2)
    hi_trend = struct1h["high_trend"]
    lo_trend = struct1h["low_trend"]

    # 6) 4H S/R map for distance filter
    highs4 = cluster_levels(swings4, "high", tolerance_pct=0.4, min_touches=4)
    lows4  = cluster_levels(swings4, "low",  tolerance_pct=0.4, min_touches=4)

    # Helper: distance to nearest 4H level on the opposite side of the trade
    def sr_distance_ok_for_short(price: float) -> bool:
        if not lows4:
            return True
        sup_level, _ = nearest_level(lows4, price, "below")
        if not sup_level:
            return True
        # require price to be at least MIN_B2_SR_ATR_MULT * ATR away from support
        dist = price - sup_level
        return dist >= MIN_B2_SR_ATR_MULT * atr1

    def sr_distance_ok_for_long(price: float) -> bool:
        if not highs4:
            return True
        res_level, _ = nearest_level(highs4, price, "above")
        if not res_level:
            return True
        dist = res_level - price
        return dist >= MIN_B2_SR_ATR_MULT * atr1

    # 7) Optional volume behaviour (light flag, not heavy distribution)
    vol = d1["vol"]
    if len(vol) >= 20:
        flag_vol_avg = float(vol.iloc[-5:-1].mean())
        prior_vol_avg = float(vol.iloc[-15:-5].mean())
        trigger_vol = float(vol.iloc[-1])
        # flag volume should be lighter than recent average,
        # trigger candle volume should not be dead
        if prior_vol_avg > 0:
            if flag_vol_avg > prior_vol_avg * 1.1:
                return None
            if trigger_vol < flag_vol_avg * 0.7:
                return None

    # =============== LONG CONTINUATION (4H uptrend) ===============
    if trend4 == "up" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        # Pullback into EMA10/21 zone, above EMA50
        in_ema_zone = (ema21_1h * 0.995 <= c <= ema10_1h * 1.005) and (c > ema50_1h)
        # Continuation MACD/RSI
        macd_ok = (macd_val > 0) and (macd1_dir == "up")
        rsi_ok = 45 <= rsi1 <= 65
        # Structural HH/HL: we don't want recent high/low trends pointing down
        struct_ok = hi_trend in ("up", "flat") and lo_trend in ("up", "flat")
        # Not too close to big 4H resistance
        sr_ok = sr_distance_ok_for_long(c)

        if in_ema_zone and macd_ok and rsi_ok and struct_ok and sr_ok:
            if not (is_bull_engulf(prev, last) or is_bull_pin(last) or is_inside_bar(prev, last)):
                return None

            entry_low = min(last["low"], ema21_1h)
            entry_high = max(last["close"], ema10_1h)
            entry_mid = (entry_low + entry_high) / 2.0

            sl_price = min(last["low"], ema50_1h * 0.997)
            price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100.0
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {
                    "side": "LONG",
                    "entry_low": entry_low,
                    "entry_high": entry_high,
                    "sl": sl_price,
                    "note": "B2_TREND_CONT · EMA pullback in 4H uptrend (mini-flag, v2.1)"
                }

    # =============== SHORT CONTINUATION (4H downtrend) ===============
    if trend4 == "down":
        # Pullback into EMA10/21 zone, below EMA50
        in_ema_zone = (ema21_1h * 1.005 >= c >= ema10_1h * 0.995) and (c < ema50_1h)
        macd_ok = (macd_val < 0) and (macd1_dir == "down")
        rsi_ok = 35 <= rsi1 <= 55
        struct_ok = hi_trend in ("down", "flat") and lo_trend in ("down", "flat")
        sr_ok = sr_distance_ok_for_short(c)

        if in_ema_zone and macd_ok and rsi_ok and struct_ok and sr_ok:
            if not (is_bear_engulf(prev, last) or is_bear_pin(last) or is_inside_bar(prev, last)):
                return None

            entry_high = max(last["high"], ema21_1h)
            entry_low = min(last["close"], ema10_1h)
            entry_mid = (entry_low + entry_high) / 2.0

            sl_price = max(last["high"], ema50_1h * 1.003)
            price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100.0
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {
                    "side": "SHORT",
                    "entry_low": entry_low,
                    "entry_high": entry_high,
                    "sl": sl_price,
                    "note": "B2_TREND_CONT · EMA pullback in 4H downtrend (mini-flag, v2.1)"
                }

    return None

# ---------- B3: Major S/R Bounce + Pattern ----------
def b3_sr_bounce_and_pattern(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    """
    B3: Major S/R bounce + pattern (double top / bottom + SFP-style).
    """
    highs = cluster_levels(swings4, "high", tolerance_pct=0.5, min_touches=3)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=0.5, min_touches=3)

    c1 = float(d1["close"].iloc[-1])
    last = d1.iloc[-1]

    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    rsi1 = float(d1["rsi"].iloc[-1])

    swings1 = detect_swings(d1["close"], look=3)
    closes = d1["close"]

    def detect_double(swings, kind: str):
        tt = "high" if kind == "top" else "low"
        idx = [i for (i, _, t) in swings if t == tt]
        if len(idx) < 2:
            return None
        a, b = idx[-2], idx[-1]
        pa, pb = float(closes.iloc[a]), float(closes.iloc[b])
        if abs(pa - pb) / pa * 100.0 > 0.35:
            return None
        return (a, b, pa, pb)

    # ----- LONG: support bounce / double bottom / SFP low -----
    if regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        if sup_level and abs(last["low"] - sup_level) / sup_level * 100 <= 0.7:
            dbl = detect_double(swings1, "bottom")
            sfp_ok = detect_sfp_low(d1, swings1)
            if (
                not is_rejection_candle(last, at_support=True, level=sup_level, tolerance_pct=0.7)
                and not dbl
                and not sfp_ok
            ):
                return None

            note_core = f"B3_SR_BOUNCE · {sup_touch}x support @ {sup_level:.4f}"
            if dbl:
                _, _, _, pb = dbl
                note_core = f"B3_PATTERN · Double Bottom near {pb:.4f}"
            if sfp_ok:
                note_core += " · SFP_low"

            if not (30 <= rsi1 <= 70 and macd1_dir in ("up", "flat")):
                return None

            entry_low = min(last["low"], sup_level)
            entry_high = max(last["close"], sup_level * 1.001)
            entry_mid = (entry_low + entry_high) / 2.0
            sl_price = min(last["low"], sup_level * 0.996)
            price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {
                    "side": "LONG",
                    "entry_low": entry_low,
                    "entry_high": entry_high,
                    "sl": sl_price,
                    "note": note_core,
                }

    # ----- SHORT: resistance rejection / double top / SFP high -----
    if res_level and abs(last["high"] - res_level) / res_level * 100 <= 0.7:
        dbl = detect_double(swings1, "top")
        sfp_ok = detect_sfp_high(d1, swings1)
        if (
            not is_rejection_candle(last, at_support=False, level=res_level, tolerance_pct=0.7)
            and not dbl
            and not sfp_ok
        ):
            return None

        note_core = f"B3_SR_BOUNCE · {res_touch}x resistance @ {res_level:.4f}"
        if dbl:
            _, _, _, pb = dbl
            note_core = f"B3_PATTERN · Double Top near {pb:.4f}"
        if sfp_ok:
            note_core += " · SFP_high"

        # RSI filter – avoid over-extended shorts like very high RSI
        if not (30 <= rsi1 <= 60 and macd1_dir in ("down", "flat")):
            return None

        entry_high = max(last["high"], res_level)
        entry_low = min(last["close"], res_level * 0.999)
        entry_mid = (entry_low + entry_high) / 2.0
        sl_price = max(last["high"], res_level * 1.004)
        price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
        if price_risk_pct <= MAX_PRICE_RISK_PCT:
            return {
                "side": "SHORT",
                "entry_low": entry_low,
                "entry_high": entry_high,
                "sl": sl_price,
                "note": note_core,
            }

    return None

# ---------- Symbol Analysis ----------
def analyze_symbol(ex, symbol: str, regime: str, dom_bias: str):
    """
    Full Bernard-style analysis for one symbol.
    Returns (signal_dict or None, raw_row dict)
    """
    d4 = fetch_tf(ex, symbol, "4h", OHLC_LIMIT_4H)
    d1 = fetch_tf(ex, symbol, "1h", OHLC_LIMIT_1H)

    # 4H EMAs
    d4["ema10"] = ema(d4.close, 10)
    d4["ema21"] = ema(d4.close, 21)
    d4["ema50"] = ema(d4.close, 50)
    d4["sma200"] = sma(d4.close, 200)
    d4["macd"], _, _ = macd(d4.close)

    # 1H EMAs
    d1["ema10"] = ema(d1.close, 10)
    d1["ema21"] = ema(d1.close, 21)
    d1["ema50"] = ema(d1.close, 50)
    d1["sma200"] = sma(d1.close, 200)
    d1["macd"], _, _ = macd(d1.close)
    d1["rsi"] = rsi(d1.close)
    d1["atr"] = atr(d1, 14)

    trend4 = classify_trend(d4.close, d4.ema10, d4.ema21, d4.ema50, d4.sma200)
    trend1 = classify_trend(d1.close, d1.ema10, d1.ema21, d1.ema50, d1.sma200)
    macd4_slope = macd_slope(d4["macd"])
    macd1_slope = macd_slope(d1["macd"])
    last_price = float(d1.close.iloc[-1])

    swings4 = detect_swings(d4.close, look=3)

    raw_row = {
        "symbol": _mk(symbol),
        "price": last_price,
        "trend4": trend4,
        "trend1": trend1,
        "regime": regime,
        "dom_bias": dom_bias,
        "macd4": macd4_slope,
        "macd1": macd1_slope,
        "note": "no setup",
    }

    signal = None

    # ---- B1: Break + Retest (highest priority)
    sig_b1 = b1_break_retest(symbol, d4, d1, swings4, trend4, regime, dom_bias)
    if sig_b1:
        signal = sig_b1
        raw_row["note"] = sig_b1["note"]

    # ---- B2: Trend continuation
     if signal is None:
        sig_b2 = b2_trend_continuation(symbol, d4, d1, swings4, trend4, regime, dom_bias)
        if sig_b2:
            signal = sig_b2
            raw_row["note"] = sig_b2["note"]

    # ---- B3: S/R bounce + pattern
    if signal is None:
        sig_b3 = b3_sr_bounce_and_pattern(symbol, d4, d1, swings4, trend4, regime, dom_bias)
        if sig_b3:
            signal = sig_b3
            raw_row["note"] = sig_b3["note"]

    # Convert to signal dict for Worker (with R-based TP)
    if signal:
        side = signal["side"]
        entry_low = float(signal["entry_low"])
        entry_high = float(signal["entry_high"])
        entry_mid = (entry_low + entry_high) / 2.0
        sl = float(signal["sl"])

        # Risk in price terms
        if side == "LONG":
            risk = entry_mid - sl
        else:
            risk = sl - entry_mid

        if risk <= 0 or not math.isfinite(risk):
            return None, raw_row

        # TP = 1R / 2R / 3R
        if side == "LONG":
            tp1 = entry_mid + risk
            tp2 = entry_mid + 2 * risk
            tp3 = entry_mid + 3 * risk
        else:
            tp1 = entry_mid - risk
            tp2 = entry_mid - 2 * risk
            tp3 = entry_mid - 3 * risk

        atr_val = d1["atr"].iloc[-1]
        atr_pct = atr_val / last_price * 100.0 if atr_val and atr_val > 0 else None
        sl_pct = abs(sl - entry_mid) / entry_mid * 100.0

        signal_dict = {
            "symbol": _mk(symbol),
            "side": side,
            "tf": "1h",
            "price": entry_mid,
            "entry": [fnum(entry_low), fnum(entry_high)],
            "entry_mid": fnum(entry_mid),
            "sl": fnum(sl),
            "tp1": fnum(tp1),
            "tp2": fnum(tp2),
            "tp3": fnum(tp3),
            "note": signal["note"],
            # telemetry for Supabase (can be null)
            "macd_slope": macd1_slope,
            "rsi": fnum(d1["rsi"].iloc[-1]),
            "atr1h_pct": fnum(atr_pct),
            "body_break_pct": None,
            "sl_max_pct": fnum(sl_pct),
            "momentum_score": None,
            "vol_spike": None,
            "volume_slope": None,
        }

        raw_row["side"] = side
        raw_row["signal"] = "TRIGGER"
        raw_row["sl"] = sl
        raw_row["tp1"] = tp1
        raw_row["tp2"] = tp2
        raw_row["tp3"] = tp3

        return signal_dict, raw_row

    return None, raw_row


# ---------- Worker Push ----------
def push_signals(signals):
    if not ALERT_WEBHOOK_URL or not signals:
        return
    url = ALERT_WEBHOOK_URL
    params = {"t": PASS_KEY}
    payload = {
        "summary": f"{len(signals)} signal(s) — Bernard System v2.3",
        "signals": signals,
    }
    try:
        with httpx.Client(timeout=10) as cli:
            r = cli.post(url, params=params, json=payload)
            print(f"[PUSH] {url} -> {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[PUSH ERROR] {e}")


# ---------- Timestamp helper ----------
def dual_ts():
    utc = datetime.now(timezone.utc)
    kl = utc.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
    return f"{utc:%Y-%m-%d %H:%M:%S UTC} | {kl:%Y-%m-%d %H:%M:%S KL}"


# ---------- Tick Once ----------
def tick_once():
    try:
        ex_price = _get_exchange(PRICE_EXCHANGE)
        ex_price.load_markets()
    except Exception as e:
        return {"engine": "bernard_v2.3", "error": str(e)}

    # dominance exchange may be same or different
    ex_dom = None
    if DOM_BTCDOM_SYMBOL and DOM_USDTDOM_SYMBOL:
        try:
            ex_dom = _get_exchange(DOM_EXCHANGE)
            ex_dom.load_markets()
        except Exception as e:
            log(f"[DOM_EXCHANGE] error: {e}")
            ex_dom = None

    regime, dom_bias = analyze_btc_regime(ex_price, ex_dom)

    signals = []
    raw = []

    for sym in WATCHLIST:
        try:
            sig, info = analyze_symbol(ex_price, sym, regime, dom_bias)
            raw.append(info)
            if sig:
                # extra guard: block fresh longs when BTC is clearly risk_off
                if regime == "risk_off" and sig["side"] == "LONG":
                    info["note"] = (info.get("note", "") or "") + " · blocked_by_risk_off"
                else:
                    signals.append(sig)
        except Exception as e:
            raw.append({"symbol": _mk(sym), "error": str(e)})

    if signals:
        push_signals(signals)

    return {
        "engine": "bernard_v2.3",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": raw,
        "note": dual_ts(),
    }
