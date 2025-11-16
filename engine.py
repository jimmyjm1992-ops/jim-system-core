# ==========================================================
#  engine.py — Bernard System v2.4 (Pure Bernard Logic)
#
#  - 3 Bernard families (now tagged S1..S5 for notes):
#       B1/S3/S4:
#           • S3: Pure Breakout/Breakdown (body + volume, no retest)
#           • S4: Breakout + Retest at strong 4H S/R
#       B2: Trend Continuation (EMA pullback / mini-flag)
#       B3/S1/S2: Major S/R Bounce + Pattern + Trendline
#           • S1: 4H S/R bounce + pattern + StochRSI
#           • S2: 4H Trendline wick-to-wick bounce + StochRSI (hardened)
#           • (S5 is execution hint only; tagged in note)
#
#  - BTC regime filter (risk_on / risk_off / neutral)
#  - Optional dominance bias (BTCDOM + USDTDOM via env)
#  - Structure-based SL, price risk capped at 5%
#  - TP = 1R / 2R / 3R
#  - Safe for Koyeb free plan (4H + 1H; optional 15m micro)
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
OHLC_LIMIT_15M  = int(os.getenv("OHLC_LIMIT_15M", "300"))

PRICE_EXCHANGE  = os.getenv("PRICE_EXCHANGE", "mexc").lower()

# Optional dominance envs (safe defaults)
DOM_EXCHANGE        = os.getenv("DOM_EXCHANGE", PRICE_EXCHANGE).lower()
DOM_BTCDOM_SYMBOL   = os.getenv("DOM_BTCDOM_SYMBOL", "").strip()
DOM_USDTDOM_SYMBOL  = os.getenv("DOM_USDTDOM_SYMBOL", "").strip()

# Optional micro-confirm (15m)
USE_15M_MICRO       = int(os.getenv("USE_15M_MICRO", "0"))

# S2 hardening (override via env if needed)
S2_MIN_ATR_PULLBACK_MULT  = float(os.getenv("S2_MIN_ATR_PULLBACK_MULT", "1.0"))  # need >= 1 ATR pullback from recent extreme before TL touch
S2_REJECTION_ATR_BODY_PCT = float(os.getenv("S2_REJECTION_ATR_BODY_PCT", "20.0"))# min body% of ATR on rejection candle
S2_VOL_SPIKE_MULT         = float(os.getenv("S2_VOL_SPIKE_MULT", "1.10"))        # vol spike multiplier (vs avg20)
S2_MACD_ZERO_LOOKBACK     = int(os.getenv("S2_MACD_ZERO_LOOKBACK", "30"))        # MACD zero-line rollover lookback
S2_MIDRANGE_REJECT_PCT    = float(os.getenv("S2_MIDRANGE_REJECT_PCT", "30.0"))   # reject if price sits in middle 40% of 20-bar range
TL_TOUCH_TOL_PCT          = float(os.getenv("TL_TOUCH_TOL_PCT", "0.4"))          # % tolerance around TL for "touch"

# StochRSI config (applied to RSI series)
STOCH_KLEN  = int(os.getenv("STOCH_KLEN", "14"))  # window for %K (on RSI)
STOCH_DLEN  = int(os.getenv("STOCH_DLEN", "3"))   # smoothing for %D

WATCHLIST = [
    s.strip()
    for s in os.getenv(
        "WATCHLIST",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,UNIUSDT,AAVEUSDT,"
        "ENAUSDT,TAOUSDT,PENDLEUSDT,VIRTUALUSDT,HBARUSDT,BCHUSDT,AVAXUSDT,"
        "LTCUSDT,DOTUSDT,SUIUSDT,NEARUSDT,OPUSDT,ARBUSDT,APTUSDT,ATOMUSDT,"
        "ALGOUSDT,INJUSDT,MATICUSDT,FTMUSDT,GALAUSDT,XLMUSDT,HYPEUSDT"
    ).split(",")
    if s.strip()
]

# Max allowed price risk (in %) between entry_mid and SL
MAX_PRICE_RISK_PCT = 5.0

# Minimum distance (in ATR multiples) from B2 entry price to nearest 4H S/R
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


# ---------- StochRSI (on RSI series) ----------
def stoch_rsi_series(rsi_series: pd.Series, klen: int = 14, dlen: int = 3):
    """Compute StochRSI %K/%D from an RSI series."""
    if len(rsi_series) < max(klen, dlen) + 5:
        k = pd.Series([50.0] * len(rsi_series), index=rsi_series.index)
        d = k.copy()
        return k, d
    low_k = rsi_series.rolling(klen).min()
    high_k = rsi_series.rolling(klen).max()
    denom = (high_k - low_k).replace(0, np.nan)
    k = ((rsi_series - low_k) / denom) * 100.0
    k = k.fillna(50.0)
    d = k.rolling(dlen).mean().fillna(method="bfill").fillna(50.0)
    return k, d


def stoch_cross_up(k: pd.Series, d: pd.Series, oversold: float = 20.0) -> bool:
    """%K crosses up through %D from oversold on the last candle."""
    if len(k) < 2 or len(d) < 2:
        return False
    return (k.iloc[-2] < d.iloc[-2] and k.iloc[-2] < oversold) and (k.iloc[-1] > d.iloc[-1])


def stoch_cross_down(k: pd.Series, d: pd.Series, overbought: float = 80.0) -> bool:
    """%K crosses down through %D from overbought on the last candle."""
    if len(k) < 2 or len(d) < 2:
        return False
    return (k.iloc[-2] > d.iloc[-2] and k.iloc[-2] > overbought) and (k.iloc[-1] < d.iloc[-1])


# ---------- Swings & S/R ----------
def detect_swings(series: pd.Series, look: int = 3):
    """
    Return list of (idx, price, type) where type in {'high','low'}.
    """
    swings = []
    vals = series.values
    for i in range(look, len(series) - look):
        window = vals[i - look: i + look + 1]
        mid = vals[i]
        if mid == window.max():
            swings.append((i, mid, "high"))
        elif mid == window.min():
            swings.append((i, mid, "low"))
    return swings


def swing_structure_1h(close: pd.Series, look: int = 2):
    """
    Read 1H swing structure for B2:
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


# ---------- Candlestick helpers ----------
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
    po, _, _, pc, _, _, _, _, _ = candle_stats(prev)
    o, _, _, c, _, _, _, _, _ = candle_stats(row)
    return (pc < po) and (c > o) and (c >= po) and (o <= pc)


def is_bear_engulf(prev, row):
    po, _, _, pc, _, _, _, _, _ = candle_stats(prev)
    o, _, _, c, _, _, _, _, _ = candle_stats(row)
    return (pc > po) and (c < o) and (c <= po) and (o >= pc)


def is_bull_pin(row):
    o, h, l, c, body, rng, uw, lw, body_pct = candle_stats(row)
    return lw > 2 * body and body_pct > 15 and min(o, c) > l + rng * 0.4


def is_bear_pin(row):
    o, h, l, c, body, rng, uw, lw, body_pct = candle_stats(row)
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
        if uw / rng > 0.5:
            return False
    else:
        if c >= level:
            return False
        if lw / rng > 0.5:
            return False
    return True


def bullish_retest_candle(row, level: float, tolerance_pct: float = 0.4):
    o, h, l, c, _, _, _, lw, _ = candle_stats(row)
    if l <= level * (1 + tolerance_pct / 100) and h >= level * (1 - tolerance_pct / 100):
        if c > o and c > level and lw > 0:
            return True
    return False


def bearish_retest_candle(row, level: float, tolerance_pct: float = 0.4):
    o, h, l, c, _, _, uw, _, _ = candle_stats(row)
    if h >= level * (1 - tolerance_pct / 100) and l <= level * (1 + tolerance_pct / 100):
        if c < o and c < level and uw > 0:
            return True
    return False


def is_rejection_candle(row, at_support: bool, level: float, tolerance_pct: float = 0.5):
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
    return h > prev_high * 1.0005 and c < prev_high


def detect_sfp_low(d1: pd.DataFrame, swings, lookback: int = 20):
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


# ---------- Volume helper ----------
def vol_spike(vol_series: pd.Series, idx: int, window: int = 20, mult: float = 1.10) -> bool:
    if idx is None:
        idx = len(vol_series) - 1
    if idx < 1 or len(vol_series) < window + 5:
        return False
    v = float(vol_series.iloc[idx])
    base = float(vol_series.iloc[max(0, idx - window): idx].mean())
    if base <= 0:
        return False
    return v >= base * mult


# ---------- Trendline helpers (wick-to-wick, simple 2-point) ----------
def trendline_value_at_current(df4: pd.DataFrame, mode: str = "up"):
    """
    Very simple TL projection (wick-to-wick on 4H closes):
      - For 'up': use last two swing lows (close-based)
      - For 'down': use last two swing highs
    Return projected price at "next" bar index as scalar (float) or None if not enough points.
    """
    swings = detect_swings(df4["close"], look=3)
    if not swings or len(swings) < 2:
        return None, 0
    if mode == "up":
        pts = [(i, p) for (i, p, t) in swings if t == "low"]
    else:
        pts = [(i, p) for (i, p, t) in swings if t == "high"]
    if len(pts) < 2:
        return None, 0
    # count "touches"
    touches = len(pts)
    a_idx, a_p = pts[-2]
    b_idx, b_p = pts[-1]
    if b_idx == a_idx:
        return None, touches
    slope = (b_p - a_p) / (b_idx - a_idx)
    # project to next index (one 4H step ahead of last candle)
    proj_idx = len(df4)  # one after last index
    tl_price = b_p + slope * (proj_idx - b_idx)
    return float(tl_price), touches


def is_trendline_touch_1h(d1: pd.DataFrame, tl_price: float, side: str, tol_pct: float) -> bool:
    """Check last 1H candle touches projected TL within tolerance."""
    if tl_price is None or len(d1) < 1:
        return False
    last = d1.iloc[-1]
    if side == "LONG":
        # wick/tag below within tolerance and close green preferred
        return (last["low"] <= tl_price * (1 + tol_pct / 100.0)) and (last["high"] >= tl_price * (1 - tol_pct / 100.0))
    else:
        return (last["high"] >= tl_price * (1 - tol_pct / 100.0)) and (last["low"] <= tl_price * (1 + tol_pct / 100.0))


def macd_zero_cross_recent(macd_line: pd.Series, lookback: int) -> str:
    """
    Return 'up' if crossed from <0 to >0 within lookback,
           'down' if crossed from >0 to <0 within lookback,
           'none' otherwise.
    """
    if len(macd_line) < lookback + 2:
        return "none"
    recent = macd_line.iloc[-(lookback+1):]
    prev = float(recent.iloc[0])
    for v in recent.iloc[1:]:
        if prev <= 0 and v > 0:
            return "up"
        if prev >= 0 and v < 0:
            return "down"
        prev = float(v)
    return "none"


def not_midrange(close: pd.Series, side: str, lookback: int = 20, midband_pct: float = 30.0) -> bool:
    """
    Reject if price sits in the middle band of recent range (sideways grind).
    For SHORT: reject if price is in lower mid-band (near lows) -> True means OK if NOT near lows.
    For LONG: reject if near highs.
    """
    if len(close) < lookback + 2:
        return True
    window = close.iloc[-lookback:]
    lo = float(window.min())
    hi = float(window.max())
    if hi <= lo:
        return True
    last = float(close.iloc[-1])
    band = (hi - lo)
    low_band_hi = lo + band * (midband_pct / 100.0)
    high_band_lo = hi - band * (midband_pct / 100.0)
    if side == "SHORT":
        # reject if sitting near lows (i.e., below low_band_hi)
        return last > low_band_hi
    else:
        # reject if sitting near highs (i.e., above high_band_lo)
        return last < high_band_lo


def pullback_from_extreme_ok(close: pd.Series, side: str, atr_val: float, mult: float = 1.0, lookback: int = 20) -> bool:
    """Price must have pulled back at least mult*ATR from recent extreme before TL touch."""
    if atr_val is None or atr_val <= 0 or len(close) < lookback + 2:
        return True
    recent = close.iloc[-lookback:]
    last = float(close.iloc[-1])
    if side == "SHORT":
        # ensure distance from recent low >= mult*ATR (avoid shorting into lows)
        recent_low = float(recent.min())
        return (last - recent_low) >= mult * atr_val
    else:
        recent_high = float(recent.max())
        return (recent_high - last) >= mult * atr_val


# ---------- Timeframe micro confirm (15m) ----------
def micro_ok(ex_name: str, symbol: str, side: str) -> bool:
    """Optional 15m micro confirmation: a tiny engulf/pin in the right direction."""
    if not USE_15M_MICRO:
        return True
    try:
        ex = _get_exchange(ex_name)
        ex.load_markets()
        d15 = fetch_tf(ex, symbol, "15m", OHLC_LIMIT_15M)
        if len(d15) < 5:
            return True
        prev = d15.iloc[-2]
        last = d15.iloc[-1]
        if side == "LONG":
            return is_bull_engulf(prev, last) or is_bull_pin(last) or is_inside_bar(prev, last)
        else:
            return is_bear_engulf(prev, last) or is_bear_pin(last) or is_inside_bar(prev, last)
    except Exception as e:
        log(f"[15m] micro_ok error: {e}")
        return True


# ---------- B1: Breakout + Retest (+ S3 pure break) ----------
def b1_break_retest(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    """
    B1: Breakout / Breakdown with Retest at strong 4H S/R.
    Also emits S3 (pure break) when a decisive candle with volume appears.
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

    # ----- S3 PURE BREAK (no retest; require volume spike) -----
    if atr_val and len(d1) >= 3:
        # LONG break above resistance
        if regime != "risk_off" and trend4 == "up" and dom_bias != "btc_pump_usdt_dump" and res_level:
            if float(prev["close"]) > res_level * 1.001 and float(prev["open"]) <= res_level:
                strong = is_strong_break_candle(d1.iloc[-3], prev, res_level, "up", 40, atr_val)
                if strong and vol_spike(d1["vol"], len(d1)-2, 20, 1.2):
                    entry_low = max(res_level, prev["close"])
                    entry_high = entry_low * 1.001
                    entry_mid = (entry_low + entry_high) / 2
                    sl = min(prev["low"], res_level * 0.997)
                    price_risk_pct = abs(entry_mid - sl) / entry_mid * 100.0
                    if price_risk_pct <= MAX_PRICE_RISK_PCT:
                        return {
                            "side": "LONG",
                            "entry_low": entry_low,
                            "entry_high": entry_high,
                            "sl": sl,
                            "note": f"S3_PURE_BREAK · {res_touch}x resistance @ {res_level:.4f}"
                        }

        # SHORT break below support
        if trend4 == "down" and sup_level:
            if float(prev["close"]) < sup_level * 0.999 and float(prev["open"]) >= sup_level:
                strong = is_strong_break_candle(d1.iloc[-3], prev, sup_level, "down", 40, atr_val)
                if strong and vol_spike(d1["vol"], len(d1)-2, 20, 1.2):
                    entry_high = min(sup_level, prev["close"])
                    entry_low = entry_high * 0.999
                    entry_mid = (entry_low + entry_high) / 2
                    sl = max(prev["high"], sup_level * 1.003)
                    price_risk_pct = abs(sl - entry_mid) / entry_mid * 100.0
                    if price_risk_pct <= MAX_PRICE_RISK_PCT:
                        return {
                            "side": "SHORT",
                            "entry_low": entry_low,
                            "entry_high": entry_high,
                            "sl": sl,
                            "note": f"S3_PURE_BREAK · {sup_touch}x support @ {sup_level:.4f}"
                        }

    # ----- S4 Break + Retest (original B1) -----
    # LONG
    if regime != "risk_off" and trend4 == "up" and dom_bias != "btc_pump_usdt_dump":
        if res_level:
            prev_close = float(prev["close"])
            if prev_close > res_level * 1.001 and float(prev["open"]) <= res_level:
                if not is_strong_break_candle(d1.iloc[-3], prev, res_level, "up", 40, atr_val):
                    return None
                if not (bullish_retest_candle(last, res_level) or is_bull_engulf(prev, last) or is_bull_pin(last)):
                    return None
                if not (45 <= rsi1 <= 70 and macd1_dir in ("up", "flat")):
                    return None

                entry_low = max(res_level, last["open"])
                entry_high = last["close"] * 1.0015
                entry_mid = (entry_low + entry_high) / 2.0
                sl_price = min(last["low"], res_level * 0.997)
                price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    note = f"B1_BREAK_RETEST · {res_touch}x resistance @ {res_level:.4f} · S4"
                    return {"side": "LONG", "entry_low": entry_low, "entry_high": entry_high, "sl": sl_price, "note": note}

    # SHORT
    if trend4 == "down":
        if sup_level:
            prev_close = float(prev["close"])
            if prev_close < sup_level * 0.999 and float(prev["open"]) >= sup_level:
                if not is_strong_break_candle(d1.iloc[-3], prev, sup_level, "down", 40, atr_val):
                    return None
                if not (bearish_retest_candle(last, sup_level) or is_bear_engulf(prev, last) or is_bear_pin(last)):
                    return None
                if not (30 <= rsi1 <= 60 and macd1_dir in ("down", "flat")):
                    return None

                entry_high = min(sup_level, last["open"])
                entry_low = last["close"] * 0.9985
                entry_mid = (entry_low + entry_high) / 2.0
                sl_price = max(last["high"], sup_level * 1.003)
                price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    note = f"B1_BREAK_RETEST · {sup_touch}x support @ {sup_level:.4f} · S4"
                    return {"side": "SHORT", "entry_low": entry_low, "entry_high": entry_high, "sl": sl_price, "note": note}

    return None


# ---------- B2: Trend Continuation (EMA pullback / mini-flag) ----------
def b2_trend_continuation(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    """
    B2 v2.4: Trend continuation using EMA pullback / mini-flag with extra guardrails
    to avoid weak/sideways pokes.
    """
    if trend4 not in ("up", "down"):
        return None

    if regime == "risk_off" and trend4 == "up":
        return None

    last = d1.iloc[-1]
    prev = d1.iloc[-2]

    c = float(last["close"])
    atr1 = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None
    if not atr1 or atr1 <= 0:
        return None

    # Mini-flag consolidation (last 4)
    recent = d1.iloc[-5:-1]
    if len(recent) < 4:
        return None

    ranges = (recent["high"] - recent["low"]).abs()
    bodies = (recent["close"] - recent["open"]).abs()
    avg_range = float(ranges.mean())
    avg_body = float(bodies.mean())
    # want relatively small bodies vs range; reject grinding too small too
    if avg_range <= 0 or avg_body / avg_range > 0.6:
        return None

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    macd_val = float(macd_line.iloc[-1])
    rsi1 = float(d1["rsi"].iloc[-1])

    ema10_1h = float(d1["ema10"].iloc[-1])
    ema21_1h = float(d1["ema21"].iloc[-1])
    ema50_1h = float(d1["ema50"].iloc[-1])

    struct1h = swing_structure_1h(d1["close"], look=2)

    highs4 = cluster_levels(swings4, "high", tolerance_pct=0.4, min_touches=4)
    lows4  = cluster_levels(swings4, "low",  tolerance_pct=0.4, min_touches=4)

    def sr_distance_ok_for_short(price: float) -> bool:
        if not lows4:
            return True
        sup_level, _ = nearest_level(lows4, price, "below")
        if not sup_level:
            return True
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

    # Optional volume behaviour
    vol = d1["vol"]
    if len(vol) >= 22:
        flag_vol_avg = float(vol.iloc[-5:-1].mean())
        prior_vol_avg = float(vol.iloc[-15:-5].mean())
        trigger_vol = float(vol.iloc[-1])
        if prior_vol_avg > 0:
            if flag_vol_avg > prior_vol_avg * 1.1:
                return None
            if trigger_vol < flag_vol_avg * 0.7:
                return None

    # ----- LONG continuation -----
    if trend4 == "up" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        in_ema_zone = (ema21_1h * 0.995 <= c <= ema10_1h * 1.005) and (c > ema50_1h)
        macd_ok = (macd_val > 0) and (macd1_dir == "up")
        rsi_ok = 45 <= rsi1 <= 65
        struct_ok = struct1h["high_trend"] in ("up", "flat") and struct1h["low_trend"] in ("up", "flat")
        sr_ok = sr_distance_ok_for_long(c)

        # extra guardrails (reuse S2 ideas)
        not_mid = not_midrange(d1["close"], "LONG", 20, S2_MIDRANGE_REJECT_PCT)

        if in_ema_zone and macd_ok and rsi_ok and struct_ok and sr_ok and not_mid:
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
                    "note": "B2_TREND_CONT · EMA pullback in 4H uptrend (mini-flag, v2.4)"
                }

    # ----- SHORT continuation -----
    if trend4 == "down":
        in_ema_zone = (ema21_1h * 1.005 >= c >= ema10_1h * 0.995) and (c < ema50_1h)
        macd_ok = (macd_val < 0) and (macd1_dir == "down")
        rsi_ok = 35 <= rsi1 <= 55
        struct_ok = struct1h["high_trend"] in ("down", "flat") and struct1h["low_trend"] in ("down", "flat")
        sr_ok = sr_distance_ok_for_short(c)

        not_mid = not_midrange(d1["close"], "SHORT", 20, S2_MIDRANGE_REJECT_PCT)

        if in_ema_zone and macd_ok and rsi_ok and struct_ok and sr_ok and not_mid:
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
                    "note": "B2_TREND_CONT · EMA pullback in 4H downtrend (mini-flag, v2.4)"
                }

    return None


# ---------- B3: Major S/R Bounce + Pattern (+ S1 + S2) ----------
def b3_sr_bounce_and_pattern(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    """
    B3: Major S/R bounce + pattern (double top/bottom + SFP-style).
    Also hosts:
      • S1: 4H S/R bounce + StochRSI gate
      • S2: 4H Trendline bounce (wick-to-wick) + hardened filters
    """
    highs = cluster_levels(swings4, "high", tolerance_pct=0.5, min_touches=4)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=0.5, min_touches=4)

    c1 = float(d1["close"].iloc[-1])
    last = d1.iloc[-1]
    prev = d1.iloc[-2]

    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    macd_val = float(macd_line.iloc[-1])
    rsi1 = float(d1["rsi"].iloc[-1])
    atr1 = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None

    k, d = stoch_rsi_series(d1["rsi"], klen=STOCH_KLEN, dlen=STOCH_DLEN)
    stoch_long_ok  = stoch_cross_up(k, d, oversold=20.0)
    stoch_short_ok = stoch_cross_down(k, d, overbought=80.0)

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

    # ==================== S1: S/R bounce (LONG) ====================
    if trend4 != "down" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        if sup_level and abs(last["low"] - sup_level) / sup_level * 100 <= 0.7:
            dbl = detect_double(swings1, "bottom")
            sfp_ok = detect_sfp_low(d1, swings1)
            pattern_ok = (
                is_rejection_candle(last, at_support=True, level=sup_level, tolerance_pct=0.7)
                or dbl is not None
                or sfp_ok
            )
            if pattern_ok:
                struct1h = swing_structure_1h(d1["close"], look=2)
                ema21_1h = float(d1["ema21"].iloc[-1])
                ema50_1h = float(d1["ema50"].iloc[-1])
                if struct1h["low_trend"] != "down" and (last["close"] >= ema21_1h and ema21_1h >= ema50_1h * 0.995):
                    # Momentum + StochRSI
                    if (macd1_dir == "up" and macd_val >= 0 and 38 <= rsi1 <= 60 and stoch_long_ok and
                        micro_ok(PRICE_EXCHANGE, symbol, "LONG")):
                        entry_low = min(last["low"], sup_level)
                        entry_high = max(last["close"], sup_level * 1.001)
                        entry_mid = (entry_low + entry_high) / 2.0
                        sl_price = min(last["low"], sup_level * 0.996)
                        price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
                        if price_risk_pct <= MAX_PRICE_RISK_PCT:
                            note_core = f"B3_SR_BOUNCE · {sup_touch}x support @ {sup_level:.4f} · S1"
                            if dbl:
                                _, _, _, pb = dbl
                                note_core = f"B3_PATTERN · Double Bottom near {pb:.4f} · S1"
                            if sfp_ok:
                                note_core += " · SFP_low"
                            return {
                                "side": "LONG",
                                "entry_low": entry_low,
                                "entry_high": entry_high,
                                "sl": sl_price,
                                "note": note_core,
                            }

    # ==================== S1: S/R bounce (SHORT) ====================
    if res_level and abs(last["high"] - res_level) / res_level * 100 <= 0.7 and trend4 != "up":
        dbl = detect_double(swings1, "top")
        sfp_ok = detect_sfp_high(d1, swings1)
        pattern_ok = (
            is_rejection_candle(last, at_support=False, level=res_level, tolerance_pct=0.7)
            or dbl is not None
            or sfp_ok
        )
        if pattern_ok:
            struct1h = swing_structure_1h(d1["close"], look=2)
            if struct1h["high_trend"] != "up":
                if (macd1_dir in ("down", "flat") and macd_val <= 0 and 40 <= rsi1 <= 65 and stoch_short_ok and
                    micro_ok(PRICE_EXCHANGE, symbol, "SHORT")):
                    entry_high = max(last["high"], res_level)
                    entry_low = min(last["close"], res_level * 0.999)
                    entry_mid = (entry_low + entry_high) / 2.0
                    sl_price = max(last["high"], res_level * 1.004)
                    price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
                    if price_risk_pct <= MAX_PRICE_RISK_PCT:
                        note_core = f"B3_SR_BOUNCE · {res_touch}x resistance @ {res_level:.4f} · S1"
                        if dbl:
                            _, _, _, pb = dbl
                            note_core = f"B3_PATTERN · Double Top near {pb:.4f} · S1"
                        if sfp_ok:
                            note_core += " · SFP_high"
                        return {
                            "side": "SHORT",
                            "entry_low": entry_low,
                            "entry_high": entry_high,
                            "sl": sl_price,
                            "note": note_core,
                        }

    # ==================== S2: Trendline bounce (hardened) ====================
    # LONG S2
    if trend4 != "down" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        tl_up, tl_touches = trendline_value_at_current(d4, mode="up")
        if tl_up:
            if is_trendline_touch_1h(d1, tl_up, "LONG", TL_TOUCH_TOL_PCT):
                # strong rejection candle vs ATR
                if atr1 and atr1 > 0:
                    o, h, l, c, body, rng, uw, lw, _ = candle_stats(last)
                    body_atr_pct = body / atr1 * 100.0 if atr1 else 0.0
                    if body_atr_pct < S2_REJECTION_ATR_BODY_PCT:
                        return None
                # MACD rollover from below zero recently
                if macd_zero_cross_recent(macd_line, S2_MACD_ZERO_LOOKBACK) not in ("up",):
                    return None
                # must pull back at least 1 ATR from recent high before TL touch
                if not pullback_from_extreme_ok(d1["close"], "LONG", atr1, S2_MIN_ATR_PULLBACK_MULT, 20):
                    return None
                # Not near highs (avoid longing into highs)
                if not not_midrange(d1["close"], "LONG", 20, S2_MIDRANGE_REJECT_PCT):
                    return None
                # Volume confirm on rejection candle
                if not vol_spike(d1["vol"], len(d1)-1, 20, S2_VOL_SPIKE_MULT):
                    return None
                # Pattern: engulf/pin/inside
                if not (is_bull_engulf(prev, last) or is_bull_pin(last) or is_inside_bar(prev, last)):
                    return None
                # StochRSI + micro
                if not (stoch_long_ok and micro_ok(PRICE_EXCHANGE, symbol, "LONG")):
                    return None

                entry_low = min(last["low"], tl_up)
                entry_high = max(last["close"], tl_up * 1.001)
                entry_mid = (entry_low + entry_high) / 2.0
                sl_price = min(last["low"], tl_up * 0.996)
                price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    return {
                        "side": "LONG",
                        "entry_low": entry_low,
                        "entry_high": entry_high,
                        "sl": sl_price,
                        "note": f"S2_TRENDLINE_BOUNCE · {tl_touches}x wick-to-wick (4H)"
                    }

    # SHORT S2
    if trend4 != "up":
        tl_dn, tl_touches = trendline_value_at_current(d4, mode="down")
        if tl_dn:
            if is_trendline_touch_1h(d1, tl_dn, "SHORT", TL_TOUCH_TOL_PCT):
                if atr1 and atr1 > 0:
                    o, h, l, c, body, rng, uw, lw, _ = candle_stats(last)
                    body_atr_pct = body / atr1 * 100.0 if atr1 else 0.0
                    if body_atr_pct < S2_REJECTION_ATR_BODY_PCT:
                        return None
                if macd_zero_cross_recent(macd_line, S2_MACD_ZERO_LOOKBACK) not in ("down",):
                    return None
                # must pull back at least 1 ATR from recent low (avoid short into lows)
                if not pullback_from_extreme_ok(d1["close"], "SHORT", atr1, S2_MIN_ATR_PULLBACK_MULT, 20):
                    return None
                # Not near lows for short
                if not not_midrange(d1["close"], "SHORT", 20, S2_MIDRANGE_REJECT_PCT):
                    return None
                if not vol_spike(d1["vol"], len(d1)-1, 20, S2_VOL_SPIKE_MULT):
                    return None
                if not (is_bear_engulf(prev, last) or is_bear_pin(last) or is_inside_bar(prev, last)):
                    return None
                if not (stoch_short_ok and micro_ok(PRICE_EXCHANGE, symbol, "SHORT")):
                    return None

                entry_high = max(last["high"], tl_dn)
                entry_low = min(last["close"], tl_dn * 0.999)
                entry_mid = (entry_low + entry_high) / 2.0
                sl_price = max(last["high"], tl_dn * 1.004)
                price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    return {
                        "side": "SHORT",
                        "entry_low": entry_low,
                        "entry_high": entry_high,
                        "sl": sl_price,
                        "note": f"S2_TRENDLINE_BOUNCE · {tl_touches}x wick-to-wick (4H)"
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

    # ---- B1: Break + Retest / Pure Break (highest priority)
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

    # ---- B3: S/R bounce + pattern + TL
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
        "summary": f"{len(signals)} signal(s) — Bernard System v2.4",
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
        return {"engine": "bernard_v2.4", "error": str(e)}

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
        "engine": "bernard_v2.4",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": raw,
        "note": dual_ts(),
    }
