# ==========================================================
#  engine.py — Bernard System v2.3 (Final Framework Tally) — POLISHED
#
#  Strategies (as per Framework):
#    S1: Support/Resistance + StochRSI   -> in B3 (S/R bounce + 1H StochRSI gate)
#    S2: Trendline + StochRSI            -> 4H wick-to-wick trendline + 1H touch + pattern + StochRSI (in B3)
#    S3: Pure Breakout/Breakdown         -> no retest, strong body + volume spike (in B1 path)
#    S4: Breakout + Retest               -> B1
#    S5: 50/50 Breakout+Retest           -> hint in `note` (no schema change)
#
#  Timeframes:
#    - 4H: analysis (S/R, trendlines, structure, EMA stack, MACD dir)
#    - 1H: triggers (patterns, StochRSI, entries, SL/TP)
#    - 15M: optional micro-confirmation (OFF by default)
#
#  Pipeline Compatibility:
#    - Same tick_once() envelope, signal_dict schema, webhook payload
#    - Safe for Koyeb free plan; 15M fetches are cached & reuse one ccxt client per tick
#    - Dual-mode webhook auth (query param + Bearer header)
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
OHLC_LIMIT_15M  = int(os.getenv("OHLC_LIMIT_15M", "400"))

PRICE_EXCHANGE  = os.getenv("PRICE_EXCHANGE", "mexc").lower()

# Optional dominance envs (safe defaults)
DOM_EXCHANGE        = os.getenv("DOM_EXCHANGE", PRICE_EXCHANGE).lower()
DOM_BTCDOM_SYMBOL   = os.getenv("DOM_BTCDOM_SYMBOL", "").strip()
DOM_USDTDOM_SYMBOL  = os.getenv("DOM_USDTDOM_SYMBOL", "").strip()

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

# ----- Risk & tolerances -----
MAX_PRICE_RISK_PCT = float(os.getenv("MAX_PRICE_RISK_PCT", "5.0"))
MIN_B2_SR_ATR_MULT = float(os.getenv("MIN_B2_SR_ATR_MULT", "1.5"))
SR_TOL_PCT         = float(os.getenv("SR_TOL_PCT", "0.4"))   # cluster tolerance
RETEST_TOL_PCT     = float(os.getenv("RETEST_TOL_PCT", "0.6"))
REJECT_TOL_PCT     = float(os.getenv("REJECT_TOL_PCT", "0.7"))

# ----- StochRSI (Framework S1/S2) -----
USE_STOCH_RSI       = int(os.getenv("USE_STOCH_RSI", "1"))  # ON by default
STOCH_RSI_LEN       = int(os.getenv("STOCH_RSI_LEN", "14")) # RSI length that feeds StochRSI
STOCH_KLEN          = int(os.getenv("STOCH_KLEN", "14"))    # %K window (on RSI)
STOCH_DLEN          = int(os.getenv("STOCH_DLEN", "3"))     # %D smoothing
STOCH_OVERSOLD      = float(os.getenv("STOCH_OVERSOLD", "20"))
STOCH_OVERBOUGHT    = float(os.getenv("STOCH_OVERBOUGHT", "80"))

# ----- Trendline (Framework S2) -----
USE_TRENDLINE_RULE  = int(os.getenv("USE_TRENDLINE_RULE", "1"))
TL_TOUCH_TOL_PCT    = float(os.getenv("TL_TOUCH_TOL_PCT", "0.5"))

# ----- 15M Micro-Confirmation (optional in framework) -----
USE_15M_MICRO       = int(os.getenv("USE_15M_MICRO", "0"))  # OFF by default
MICRO_USE_STOCH     = int(os.getenv("MICRO_USE_STOCH", "1"))
MICRO_USE_PATTERN   = int(os.getenv("MICRO_USE_PATTERN", "1"))

# ----- Pure Breakout (Framework S3) -----
ENABLE_PURE_BREAKOUT  = int(os.getenv("ENABLE_PURE_BREAKOUT", "1"))
BREAK_BODY_ATR_MIN    = float(os.getenv("BREAK_BODY_ATR_MIN", "40"))
BREAK_VOL_SPIKE_MULT  = float(os.getenv("BREAK_VOL_SPIKE_MULT", "1.2"))

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
    try:
        a = float(a); b = float(b)
    except Exception:
        return 1e9
    mid = (a + b) / 2.0
    if mid <= 0:
        return 1e9
    return abs(a - b) / mid * 100.0

def _mk(symbol: str) -> str:
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

# ---------- Indicators ----------
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

def stoch_rsi_series(rsi_series: pd.Series, klen=14, dlen=3):
    # %K = normalized RSI over klen; %D = SMA of %K over dlen
    rmin = rsi_series.rolling(klen).min()
    rmax = rsi_series.rolling(klen).max()
    denom = (rmax - rmin).replace(0, np.nan)
    k = ((rsi_series - rmin) / denom).clip(0, 1) * 100.0
    k = k.fillna(method="bfill").fillna(0.0)
    d = k.rolling(dlen).mean().fillna(method="bfill").fillna(0.0)
    return k, d

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

def stoch_cross_up(k: pd.Series, d: pd.Series, oversold=20.0) -> bool:
    if len(k) < 2 or len(d) < 2:
        return False
    return (k.iloc[-2] < d.iloc[-2] and k.iloc[-2] < oversold) and (k.iloc[-1] > d.iloc[-1])

def stoch_cross_down(k: pd.Series, d: pd.Series, overbought=80.0) -> bool:
    if len(k) < 2 or len(d) < 2:
        return False
    return (k.iloc[-2] > d.iloc[-2] and k.iloc[-2] > overbought) and (k.iloc[-1] < d.iloc[-1])

# ---------- Swings & S/R ----------
def detect_swings(series: pd.Series, look: int = 3):
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

    return {"high_trend": trend_from_idx(high_idx), "low_trend": trend_from_idx(low_idx)}

def cluster_levels(swings, level_type: str, tolerance_pct: float = SR_TOL_PCT, min_touches: int = 4):
    vals = [p for (_, p, t) in swings if t == level_type]
    if not vals:
        return []
    vals = sorted(vals)
    clusters = [[vals[0], 1]]
    for v in vals[1:]:
        avg, cnt = clusters[-1]
        if abs(v - avg) / avg * 100 <= tolerance_pct:
            new_avg = (avg * cnt + v) / (cnt + 1)
            clusters[-1] = [new_avg, cnt + 1]
        else:
            clusters.append([v, 1])
    return [(avg, cnt) for (avg, cnt) in clusters if cnt >= min_touches]

def nearest_level(levels, price: float, direction: str):
    if not levels:
        return None, 0
    if direction == "above":
        c = [(lv, n) for (lv, n) in levels if lv >= price]
        if not c:
            return None, 0
        return min(c, key=lambda x: x[0])
    else:
        c = [(lv, n) for (lv, n) in levels if lv <= price]
        if not c:
            return None, 0
        return max(c, key=lambda x: x[0])

# ---------- Trend classification ----------
def classify_trend(close: pd.Series, e10: pd.Series, e21: pd.Series, e50: pd.Series, s200: pd.Series):
    if len(close) < 200:
        return "chop"
    c = close.iloc[-1]
    v10 = e10.iloc[-1]; v21 = e21.iloc[-1]; v50 = e50.iloc[-1]; v200 = s200.iloc[-1]
    if c > v10 > v21 > v50 > v200:
        return "up"
    if c < v10 < v21 < v50 < v200:
        return "down"
    return "chop"

# ---------- Trendline helpers (4H wick-to-wick) ----------
def trendline_value_at_current(df4: pd.DataFrame, mode: str = "up"):
    """
    Build a simple line from last two swing lows (up) or swing highs (down) on 4H.
    Returns (line_price_at_last_index, confirmed:bool) or (None, False) if not enough data.
    """
    if mode == "up":
        swings = detect_swings(df4["low"], look=3)
        pts = [(i, float(df4["low"].iloc[i])) for (i, p, t) in swings if t == "low"]
    else:
        swings = detect_swings(df4["high"], look=3)
        pts = [(i, float(df4["high"].iloc[i])) for (i, p, t) in swings if t == "high"]

    if len(pts) < 2:
        return None, False

    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    if x2 == x1:
        return None, False
    m = (y2 - y1) / (x2 - x1)
    x_cur = len(df4) - 1
    y_cur = y1 + m * (x_cur - x1)
    confirmed = len(pts) >= 3
    return y_cur, confirmed

def is_trendline_touch_1h(d1: pd.DataFrame, tl_price: float, side: str, tol_pct: float = TL_TOUCH_TOL_PCT):
    if tl_price is None or len(d1) < 1:
        return False
    last = d1.iloc[-1]
    if side == "LONG":
        return last["low"] <= tl_price * (1 + tol_pct / 100.0)
    else:
        return last["high"] >= tl_price * (1 - tol_pct / 100.0)

# ---------- BTC Regime + Dominance ----------
def analyze_btc_regime(ex_price, ex_dom=None):
    regime = "neutral"; dom_bias = "neutral"
    try:
        df4 = fetch_tf(ex_price, "BTCUSDT", "4h", OHLC_LIMIT_4H)
        df1 = fetch_tf(ex_price, "BTCUSDT", "1h", OHLC_LIMIT_1H)

        df4["ema10"] = ema(df4.close, 10); df4["ema21"] = ema(df4.close, 21)
        df4["ema50"] = ema(df4.close, 50); df4["sma200"] = sma(df4.close, 200)
        df4["macd"], _, _ = macd(df4.close)

        df1["ema10"] = ema(df1.close, 10); df1["ema21"] = ema(df1.close, 21)
        df1["ema50"] = ema(df1.close, 50); df1["sma200"] = sma(df1.close, 200)
        df1["macd"], _, _ = macd(df1.close); df1["rsi"] = rsi(df1.close)

        trend4 = classify_trend(df4.close, df4.ema10, df4.ema21, df4.ema50, df4.sma200)
        trend1 = classify_trend(df1.close, df1.ema10, df1.ema21, df1.ema50, df1.sma200)
        macd1_slope = macd_slope(df1["macd"])
        rsi1 = float(df1["rsi"].iloc[-1])

        if trend4 == "up" and trend1 in ("up","chop") and macd1_slope == "up" and rsi1 > 50:
            regime = "risk_on"
        elif trend4 == "down" and trend1 in ("down","chop") and macd1_slope == "down" and rsi1 < 50:
            regime = "risk_off"
        else:
            regime = "neutral"
    except Exception as e:
        log(f"[BTC_REGIME] error: {e}")
        regime = "neutral"

    if ex_dom is not None and DOM_BTCDOM_SYMBOL and DOM_USDTDOM_SYMBOL:
        try:
            btcd = fetch_tf(ex_dom, DOM_BTCDOM_SYMBOL, "4h", OHLC_LIMIT_4H)
            usdt = fetch_tf(ex_dom, DOM_USDTDOM_SYMBOL, "4h", OHLC_LIMIT_4H)
            btcd["ema20"] = ema(btcd.close, 20); usdt["ema20"] = ema(usdt.close, 20)
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
    o = row["open"]; h = row["high"]; l = row["low"]; c = row["close"]
    body = abs(c - o); rng = h - l
    if rng <= 0: rng = 1e-9
    uw = h - max(o, c); lw = min(o, c) - l
    body_pct = body / rng * 100.0
    return o, h, l, c, body, rng, uw, lw, body_pct

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
    Strong breakout candle (compat signature):
    - body size >= min_body_atr_pct of ATR
    - close clearly beyond level
    - wick against direction not too large
    NOTE: 'prev' is kept only for backward compatibility; logic uses 'row'.
    """
    o, h, l, c, body, rng, uw, lw, _ = candle_stats(row)
    if atr_val is None or atr_val <= 0: return False
    body_atr_pct = body / atr_val * 100.0
    if body_atr_pct < min_body_atr_pct: return False
    if direction == "up":
        if c <= level: return False
        if uw / rng > 0.5: return False
    else:
        if c >= level: return False
        if lw / rng > 0.5: return False
    return True

def bullish_retest_candle(row, level: float, tolerance_pct: float = RETEST_TOL_PCT):
    o, h, l, c, _, _, _, lw, _ = candle_stats(row)
    if l <= level * (1 + tolerance_pct / 100) and h >= level * (1 - tolerance_pct / 100):
        if c > o and c > level and lw > 0:
            return True
    return False

def bearish_retest_candle(row, level: float, tolerance_pct: float = RETEST_TOL_PCT):
    o, h, l, c, _, _, uw, _, _ = candle_stats(row)
    if h >= level * (1 - tolerance_pct / 100) and l <= level * (1 + tolerance_pct / 100):
        if c < o and c < level and uw > 0:
            return True
    return False

def is_rejection_candle(row, at_support: bool, level: float, tolerance_pct: float = REJECT_TOL_PCT):
    o, h, l, c, body, rng, uw, lw, body_pct = candle_stats(row)
    if rng <= 0: return False
    if at_support:
        if (l <= level * (1 + tolerance_pct / 100)
            and (lw / rng) > 0.4
            and c > o
            and body_pct > 20):
            return True
    else:
        if (h >= level * (1 - tolerance_pct / 100)
            and (uw / rng) > 0.4
            and c < o
            and body_pct > 20):
            return True
    return False

def vol_spike(vol: pd.Series, idx: int, window: int = 20, mult: float = 1.2) -> bool:
    if idx - window < 0: return False
    avg = float(vol.iloc[idx-window:idx].mean())
    if avg <= 0: return False
    return float(vol.iloc[idx]) >= avg * mult

# ---------- SFP ----------
def detect_sfp_high(d1: pd.DataFrame, swings, lookback: int = 20):
    if len(d1) < 5 or not swings: return False
    last_idx = len(d1) - 1
    recent_swings = [i for (i, p, t) in swings if t == "high" and last_idx - lookback <= i < last_idx]
    if not recent_swings: return False
    last = d1.iloc[-1]; h = last["high"]; c = last["close"]
    prev_idx = recent_swings[-1]; prev_high = float(d1["high"].iloc[prev_idx])
    return h > prev_high * 1.0005 and c < prev_high

def detect_sfp_low(d1: pd.DataFrame, swings, lookback: int = 20):
    if len(d1) < 5 or not swings: return False
    last_idx = len(d1) - 1
    recent_swings = [i for (i, p, t) in swings if t == "low" and last_idx - lookback <= i < last_idx]
    if not recent_swings: return False
    last = d1.iloc[-1]; l = last["low"]; c = last["close"]
    prev_idx = recent_swings[-1]; prev_low = float(d1["low"].iloc[prev_idx])
    return l < prev_low * 0.9995 and c > prev_low

# ---------- B1: Breakout + Retest (+ Pure Breakout) ----------
def b1_break_retest(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    highs = cluster_levels(swings4, "high", tolerance_pct=SR_TOL_PCT, min_touches=5)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=SR_TOL_PCT, min_touches=5)

    c1 = float(d1["close"].iloc[-1])
    prev = d1.iloc[-2]; last = d1.iloc[-1]

    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    rsi1 = float(d1["rsi"].iloc[-1])
    atr_val = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None

    # ---- LONG: breakout above resistance
    if regime != "risk_off" and trend4 == "up" and dom_bias != "btc_pump_usdt_dump" and res_level:
        prev_close = float(prev["close"])

        broke = prev_close > res_level * 1.001 and float(prev["open"]) <= res_level
        strong = is_strong_break_candle(d1.iloc[-3], prev, res_level, "up", min_body_atr_pct=BREAK_BODY_ATR_MIN, atr_val=atr_val)

        if broke and strong:
            # Strategy 4: break + retest
            retest_ok = (bullish_retest_candle(last, res_level, RETEST_TOL_PCT)
                         or is_bull_engulf(prev, last)
                         or is_bull_pin(last))
            momentum_ok = (45 <= rsi1 <= 70 and macd1_dir in ("up","flat"))

            if retest_ok and momentum_ok:
                entry_low  = max(res_level, last["open"])
                entry_high = last["close"] * 1.0015
                entry_mid  = (entry_low + entry_high) / 2.0
                sl_price   = min(last["low"], res_level * 0.997)
                price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    note = f"B1_BREAK_RETEST · {res_touch}x resistance @ {res_level:.4f} · S4"
                    note += " · S5_SPLIT_50_50"
                    return {"side":"LONG","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,"note":note}

            # Strategy 3: Pure breakout (no retest) with volume spike
            if ENABLE_PURE_BREAKOUT:
                if vol_spike(d1["vol"], idx=len(d1)-2, window=20, mult=BREAK_VOL_SPIKE_MULT):
                    entry_low  = max(res_level, prev_close)
                    entry_high = max(entry_low * 1.0015, last["close"])
                    entry_mid  = (entry_low + entry_high) / 2.0
                    sl_price   = min(prev["low"], res_level * 0.997)
                    price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
                    if price_risk_pct <= MAX_PRICE_RISK_PCT:
                        note = f"S3_PURE_BREAK · {res_touch}x resistance @ {res_level:.4f}"
                        return {"side":"LONG","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,"note":note}

    # ---- SHORT: breakdown below support
    if trend4 == "down" and sup_level:
        prev_close = float(prev["close"])
        broke = prev_close < sup_level * 0.999 and float(prev["open"]) >= sup_level
        strong = is_strong_break_candle(d1.iloc[-3], prev, sup_level, "down", min_body_atr_pct=BREAK_BODY_ATR_MIN, atr_val=atr_val)

        if broke and strong:
            retest_ok = (bearish_retest_candle(last, sup_level, RETEST_TOL_PCT)
                         or is_bear_engulf(prev, last)
                         or is_bear_pin(last))
            momentum_ok = (30 <= rsi1 <= 60 and macd1_dir in ("down","flat"))

            if retest_ok and momentum_ok:
                entry_high = min(sup_level, last["open"])
                entry_low  = last["close"] * 0.9985
                entry_mid  = (entry_low + entry_high) / 2.0
                sl_price   = max(last["high"], sup_level * 1.003)
                price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    note = f"B1_BREAK_RETEST · {sup_touch}x support @ {sup_level:.4f} · S4"
                    note += " · S5_SPLIT_50_50"
                    return {"side":"SHORT","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,"note":note}

            # Strategy 3: Pure breakdown with volume spike
            if ENABLE_PURE_BREAKOUT:
                if vol_spike(d1["vol"], idx=len(d1)-2, window=20, mult=BREAK_VOL_SPIKE_MULT):
                    entry_high = min(sup_level, prev["close"])
                    entry_low  = min(entry_high * 0.9985, last["close"])
                    entry_mid  = (entry_low + entry_high) / 2.0
                    sl_price   = max(prev["high"], sup_level * 1.003)
                    price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
                    if price_risk_pct <= MAX_PRICE_RISK_PCT:
                        note = f"S3_PURE_BREAK · {sup_touch}x support @ {sup_level:.4f}"
                        return {"side":"SHORT","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,"note":note}

    return None

# ---------- B2: Trend Continuation (EMA pullback / mini-flag) ----------
def b2_trend_continuation(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    if trend4 not in ("up","down"):
        return None
    if regime == "risk_off" and trend4 == "up":
        return None

    last = d1.iloc[-1]; prev = d1.iloc[-2]
    c = float(last["close"])
    atr1 = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None
    if not atr1 or atr1 <= 0:
        return None

    recent = d1.iloc[-5:-1]
    if len(recent) < 4:
        return None
    ranges = (recent["high"] - recent["low"]).abs()
    bodies = (recent["close"] - recent["open"]).abs()
    avg_range = float(ranges.mean()); avg_body = float(bodies.mean())
    if avg_range <= 0 or avg_body / avg_range > 0.7:
        return None

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    macd_val = float(macd_line.iloc[-1])
    rsi1 = float(d1["rsi"].iloc[-1])
    ema10_1h = float(d1["ema10"].iloc[-1]); ema21_1h = float(d1["ema21"].iloc[-1]); ema50_1h = float(d1["ema50"].iloc[-1])

    struct1h = swing_structure_1h(d1["close"], look=2)
    hi_trend = struct1h["high_trend"]; lo_trend = struct1h["low_trend"]

    highs4 = cluster_levels(swings4, "high", tolerance_pct=SR_TOL_PCT, min_touches=4)
    lows4  = cluster_levels(swings4, "low",  tolerance_pct=SR_TOL_PCT, min_touches=4)

    def sr_distance_ok_for_short(price: float) -> bool:
        if not lows4: return True
        sup_level, _ = nearest_level(lows4, price, "below")
        if not sup_level: return True
        return (price - sup_level) >= MIN_B2_SR_ATR_MULT * atr1

    def sr_distance_ok_for_long(price: float) -> bool:
        if not highs4: return True
        res_level, _ = nearest_level(highs4, price, "above")
        if not res_level: return True
        return (res_level - price) >= MIN_B2_SR_ATR_MULT * atr1

    vol = d1["vol"]
    if len(vol) >= 20:
        flag_vol_avg = float(vol.iloc[-5:-1].mean())
        prior_vol_avg = float(vol.iloc[-15:-5].mean())
        trigger_vol = float(vol.iloc[-1])
        if prior_vol_avg > 0:
            if flag_vol_avg > prior_vol_avg * 1.1:
                return None
            if trigger_vol < flag_vol_avg * 0.7:
                return None

    # LONG
    if trend4 == "up" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        in_ema_zone = (ema21_1h * 0.995 <= c <= ema10_1h * 1.005) and (c > ema50_1h)
        macd_ok = (macd_val > 0) and (macd1_dir == "up")
        rsi_ok  = 45 <= rsi1 <= 65
        struct_ok = hi_trend in ("up","flat") and lo_trend in ("up","flat")
        sr_ok = sr_distance_ok_for_long(c)

        if in_ema_zone and macd_ok and rsi_ok and struct_ok and sr_ok:
            if not (is_bull_engulf(prev, last) or is_bull_pin(last) or is_inside_bar(prev, last)):
                return None
            entry_low  = min(last["low"], ema21_1h)
            entry_high = max(last["close"], ema10_1h)
            entry_mid  = (entry_low + entry_high) / 2.0
            sl_price   = min(last["low"], ema50_1h * 0.997)
            price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100.0
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {"side":"LONG","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,
                        "note":"B2_TREND_CONT · EMA pullback in 4H uptrend (mini-flag, v2.1)"}

    # SHORT
    if trend4 == "down":
        in_ema_zone = (ema21_1h * 1.005 >= c >= ema10_1h * 0.995) and (c < ema50_1h)
        macd_ok = (macd_val < 0) and (macd1_dir == "down")
        rsi_ok  = 35 <= rsi1 <= 55
        struct_ok = hi_trend in ("down","flat") and lo_trend in ("down","flat")
        sr_ok = sr_distance_ok_for_short(c)

        if in_ema_zone and macd_ok and rsi_ok and struct_ok and sr_ok:
            if not (is_bear_engulf(prev, last) or is_bear_pin(last) or is_inside_bar(prev, last)):
                return None
            entry_high = max(last["high"], ema21_1h)
            entry_low  = min(last["close"], ema10_1h)
            entry_mid  = (entry_low + entry_high) / 2.0
            sl_price   = max(last["high"], ema50_1h * 1.003)
            price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100.0
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {"side":"SHORT","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,
                        "note":"B2_TREND_CONT · EMA pullback in 4H downtrend (mini-flag, v2.1)"}

    return None

# ---------- micro 15m: cached client + per-symbol cache (per tick) ----------
_MICRO_EX = None
_MICRO_CACHE = {}  # key: (symbol) -> df15m

def _get_micro_ex():
    global _MICRO_EX
    if _MICRO_EX is None:
        _MICRO_EX = _get_exchange(PRICE_EXCHANGE)
        try:
            _MICRO_EX.load_markets()
        except Exception as e:
            log(f"[15M] load_markets error: {e}")
    return _MICRO_EX

def _get_15m_df(symbol: str):
    if symbol in _MICRO_CACHE:
        return _MICRO_CACHE[symbol]
    ex = _get_micro_ex()
    try:
        df15 = fetch_tf(ex, symbol, "15m", OHLC_LIMIT_15M)
        _MICRO_CACHE[symbol] = df15
        return df15
    except Exception as e:
        log(f"[15M] fetch error: {e}")
        return None

# ---------- B3: Major S/R Bounce + Pattern (+ Trendline + StochRSI) ----------
def b3_sr_bounce_and_pattern(symbol, d4, d1, swings4, trend4, regime, dom_bias):
    highs = cluster_levels(swings4, "high", tolerance_pct=SR_TOL_PCT, min_touches=4)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=SR_TOL_PCT, min_touches=4)

    c1 = float(d1["close"].iloc[-1])
    last = d1.iloc[-1]
    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    macd_line, _, _ = macd(d1["close"])
    macd1_dir = macd_slope(macd_line)
    macd_val = float(macd_line.iloc[-1])  # FIXED
    rsi1 = float(d1["rsi"].iloc[-1])
    atr1 = float(d1["atr"].iloc[-1]) if "atr" in d1.columns else None

    # StochRSI (1H)
    if USE_STOCH_RSI:
        stoch_k = d1["stoch_k"]; stoch_d = d1["stoch_d"]
        stoch_long_ok  = stoch_cross_up(stoch_k, stoch_d, STOCH_OVERSOLD)
        stoch_short_ok = stoch_cross_down(stoch_k, stoch_d, STOCH_OVERBOUGHT)
    else:
        stoch_long_ok = stoch_short_ok = True

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

    # Optional: 15M micro-confirmation (cached)
    def micro_ok(side: str) -> bool:
        if not USE_15M_MICRO:
            return True
        try:
            d15 = _get_15m_df(symbol)
            if d15 is None or len(d15) < 2:
                return True  # fail-open to preserve signals
            ok1 = True; ok2 = True
            if MICRO_USE_STOCH:
                rsi15 = rsi(d15["close"], STOCH_RSI_LEN)
                k15, d15s = stoch_rsi_series(rsi15, STOCH_KLEN, STOCH_DLEN)
                if side == "LONG":
                    ok1 = stoch_cross_up(k15, d15s, STOCH_OVERSOLD)
                else:
                    ok1 = stoch_cross_down(k15, d15s, STOCH_OVERBOUGHT)
            if MICRO_USE_PATTERN:
                p, l = d15.iloc[-2], d15.iloc[-1]
                if side == "LONG":
                    ok2 = is_bull_engulf(p, l) or is_bull_pin(l) or is_inside_bar(p, l)
                else:
                    ok2 = is_bear_engulf(p, l) or is_bear_pin(l) or is_inside_bar(p, l)
            return ok1 and ok2
        except Exception as e:
            log(f"[15M] micro confirm error: {e}")
            return True

    # ----- LONG path: S/R bounce (S1) OR Trendline bounce (S2) -----
    long_candidate = None
    long_note_core = None

    # S1: S/R bounce at support
    if trend4 != "down" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        if sup_level and abs(last["low"] - sup_level) / sup_level * 100 <= REJECT_TOL_PCT:
            dbl = detect_double(swings1, "bottom")
            sfp_ok = detect_sfp_low(d1, swings1)
            if (is_rejection_candle(last, at_support=True, level=sup_level, tolerance_pct=REJECT_TOL_PCT)
                or dbl or sfp_ok):
                struct1h = swing_structure_1h(d1["close"], look=2)
                if struct1h["low_trend"] != "down":
                    ema21_1h = float(d1["ema21"].iloc[-1]); ema50_1h = float(d1["ema50"].iloc[-1])
                    if last["close"] >= ema21_1h and ema21_1h >= ema50_1h * 0.995:
                        momentum_ok = (macd1_dir == "up" and macd_val >= 0 and 38 <= rsi1 <= 60)
                        if momentum_ok and stoch_long_ok and micro_ok("LONG"):
                            long_candidate = ("SR", sup_level, sup_touch, dbl, sfp_ok)
                            note = f"B3_SR_BOUNCE · {sup_touch}x support @ {sup_level:.4f} · S1"
                            if dbl: note = f"B3_PATTERN · Double Bottom near {dbl[3]:.4f} · S1"
                            if sfp_ok: note += " · SFP_low"
                            long_note_core = note

    # S2: Trendline bounce (uptrend support line)
    if long_candidate is None and USE_TRENDLINE_RULE and trend4 != "down" and regime != "risk_off" and dom_bias != "btc_pump_usdt_dump":
        tl_val, confirmed = trendline_value_at_current(d4, mode="up")
        if tl_val and is_trendline_touch_1h(d1, tl_val, side="LONG", tol_pct=TL_TOUCH_TOL_PCT):
            prev = d1.iloc[-2]; last = d1.iloc[-1]
            if is_bull_engulf(prev, last) or is_bull_pin(last) or is_inside_bar(prev, last):
                if stoch_long_ok and micro_ok("LONG"):
                    long_candidate = ("TL", tl_val, 3 if confirmed else 2, None, None)
                    long_note_core = f"S2_TRENDLINE_BOUNCE · {'3x' if confirmed else '2x'} touches (wick-to-wick)"

    if long_candidate:
        typ, lvl, touches, dbl, sfp_ok = long_candidate
        ref_level = lvl
        entry_low  = min(last["low"], ref_level)
        entry_high = max(last["close"], ref_level * 1.001)
        entry_mid  = (entry_low + entry_high) / 2.0
        sl_price   = min(last["low"], ref_level * 0.996)
        # Optional: ensure ≥ ~1*ATR gap to 4H resistance
        if atr1 and highs:
            nearest_res, _ = nearest_level(highs, float(last["close"]), "above")
            if nearest_res and (nearest_res - float(last["close"])) < atr1:
                long_candidate = None
        if long_candidate:
            price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {"side":"LONG","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,"note": long_note_core}

    # ----- SHORT path: S/R rejection (S1 mirror) OR Trendline down (S2 mirror) -----
    short_candidate = None
    short_note_core = None

    # S1 mirror: near resistance
    if trend4 != "up":
        if res_level and abs(last["high"] - res_level) / res_level * 100 <= REJECT_TOL_PCT:
            dbl = detect_double(swings1, "top")
            sfp_ok = detect_sfp_high(d1, swings1)
            if (is_rejection_candle(last, at_support=False, level=res_level, tolerance_pct=REJECT_TOL_PCT)
                or dbl or sfp_ok):
                struct1h = swing_structure_1h(d1["close"], look=2)
                if struct1h["high_trend"] != "up":
                    momentum_ok = (macd_val <= 0 and macd1_dir == "down" and 40 <= rsi1 <= 65)
                    if momentum_ok and stoch_short_ok and micro_ok("SHORT"):
                        short_candidate = ("SR", res_level, res_touch, dbl, sfp_ok)
                        note = f"B3_SR_BOUNCE · {res_touch}x resistance @ {res_level:.4f} · S1"
                        if dbl: note = f"B3_PATTERN · Double Top near {dbl[3]:.4f} · S1"
                        if sfp_ok: note += " · SFP_high"
                        short_note_core = note

    # S2 mirror: downtrend line (resistance)
    if short_candidate is None and USE_TRENDLINE_RULE and trend4 != "up":
        tl_val, confirmed = trendline_value_at_current(d4, mode="down")
        if tl_val and is_trendline_touch_1h(d1, tl_val, side="SHORT", tol_pct=TL_TOUCH_TOL_PCT):
            prev = d1.iloc[-2]; last = d1.iloc[-1]
            if is_bear_engulf(prev, last) or is_bear_pin(last) or is_inside_bar(prev, last):
                if stoch_short_ok and micro_ok("SHORT"):
                    short_candidate = ("TL", tl_val, 3 if confirmed else 2, None, None)
                    short_note_core = f"S2_TRENDLINE_BOUNCE · {'3x' if confirmed else '2x'} touches (wick-to-wick)"

    if short_candidate:
        typ, lvl, touches, dbl, sfp_ok = short_candidate
        ref_level = lvl
        entry_high = max(last["high"], ref_level)
        entry_low  = min(last["close"], ref_level * 0.999)
        entry_mid  = (entry_low + entry_high) / 2.0
        sl_price   = max(last["high"], ref_level * 1.004)
        if atr1 and lows:
            nearest_sup, _ = nearest_level(lows, float(last["close"]), "below")
            if nearest_sup and (float(last["close"]) - nearest_sup) < atr1:
                short_candidate = None
        if short_candidate:
            price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
            if price_risk_pct <= MAX_PRICE_RISK_PCT:
                return {"side":"SHORT","entry_low":entry_low,"entry_high":entry_high,"sl":sl_price,"note": short_note_core}

    return None

# ---------- Symbol Analysis ----------
def analyze_symbol(ex, symbol: str, regime: str, dom_bias: str):
    d4 = fetch_tf(ex, symbol, "4h", OHLC_LIMIT_4H)
    d1 = fetch_tf(ex, symbol, "1h", OHLC_LIMIT_1H)

    # 4H EMAs
    d4["ema10"] = ema(d4.close, 10); d4["ema21"] = ema(d4.close, 21)
    d4["ema50"] = ema(d4.close, 50); d4["sma200"] = sma(d4.close, 200)
    d4["macd"], _, _ = macd(d4.close)

    # 1H EMAs
    d1["ema10"] = ema(d1.close, 10); d1["ema21"] = ema(d1.close, 21)
    d1["ema50"] = ema(d1.close, 50); d1["sma200"] = sma(d1.close, 200)
    d1["macd"], _, _ = macd(d1.close)

    # RSI for general logic (14) and for StochRSI feed (STOCH_RSI_LEN)
    d1["rsi"] = rsi(d1.close, 14)
    if USE_STOCH_RSI:
        rsi_for_stoch = rsi(d1.close, STOCH_RSI_LEN)
        k, d = stoch_rsi_series(rsi_for_stoch, klen=STOCH_KLEN, dlen=STOCH_DLEN)
        d1["stoch_k"] = k; d1["stoch_d"] = d
    else:
        d1["stoch_k"] = pd.Series(index=d1.index, dtype=float)
        d1["stoch_d"] = pd.Series(index=d1.index, dtype=float)

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

    # B1 (S4 and S3)
    sig_b1 = b1_break_retest(symbol, d4, d1, swings4, trend4, regime, dom_bias)
    if sig_b1:
        signal = sig_b1
        raw_row["note"] = sig_b1["note"]

    # B2 continuation
    if signal is None:
        sig_b2 = b2_trend_continuation(symbol, d4, d1, swings4, trend4, regime, dom_bias)
        if sig_b2:
            signal = sig_b2
            raw_row["note"] = sig_b2["note"]

    # B3 (S1 + S2 with StochRSI)
    if signal is None:
        sig_b3 = b3_sr_bounce_and_pattern(symbol, d4, d1, swings4, trend4, regime, dom_bias)
        if sig_b3:
            signal = sig_b3
            raw_row["note"] = sig_b3["note"]

    # Convert to signal dict
    if signal:
        side = signal["side"]
        entry_low = float(signal["entry_low"]); entry_high = float(signal["entry_high"])
        entry_mid = (entry_low + entry_high) / 2.0
        sl = float(signal["sl"])
        risk = (entry_mid - sl) if side == "LONG" else (sl - entry_mid)
        if risk <= 0 or not math.isfinite(risk):
            return None, raw_row

        if side == "LONG":
            tp1 = entry_mid + risk; tp2 = entry_mid + 2*risk; tp3 = entry_mid + 3*risk
        else:
            tp1 = entry_mid - risk; tp2 = entry_mid - 2*risk; tp3 = entry_mid - 3*risk

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
        raw_row["tp1"] = tp1; raw_row["tp2"] = tp2; raw_row["tp3"] = tp3

        return signal_dict, raw_row

    return None, raw_row

# ---------- Worker Push ----------
def push_signals(signals):
    if not ALERT_WEBHOOK_URL or not signals:
        return
    url = ALERT_WEBHOOK_URL
    params = {"t": PASS_KEY} if PASS_KEY else {}
    headers = {}
    if PASS_KEY:
        headers["Authorization"] = f"Bearer {PASS_KEY}"
    payload = {
        "summary": f"{len(signals)} signal(s) — Bernard System v2.3",
        "signals": signals,
    }
    try:
        with httpx.Client(timeout=10) as cli:
            r = cli.post(url, params=params, headers=headers, json=payload)
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

    # reset micro cache each tick
    global _MICRO_EX, _MICRO_CACHE
    _MICRO_EX = None
    _MICRO_CACHE = {}

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
    seen_ids = set()  # de-dupe within tick

    for sym in WATCHLIST:
        try:
            sig, info = analyze_symbol(ex_price, sym, regime, dom_bias)
            raw.append(info)
            if sig:
                if regime == "risk_off" and sig["side"] == "LONG":
                    info["note"] = (info.get("note", "") or "") + " · blocked_by_risk_off"
                else:
                    k = f"{sig['symbol']}|{sig['side']}|{round(sig['entry_mid'] or sig['price'], 6)}|{round(sig['sl'] or 0, 6)}"
                    if k in seen_ids:
                        continue
                    seen_ids.add(k)
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

