# ==========================================================
#  engine.py — JIM Framework v6.1 (Bernard-style)
#  - Priority: Trend + Structure + Pattern  > Indicators
#  - 3 tiers:
#       T1: Breakout + Retest (core Bernard)
#       T2: S/R or Trendline Bounce + Candle confirmation
#       T3: Chart Pattern (Double Top / Bottom)
#  - BTC regime filter (risk-on / risk-off / neutral)
#  - Structure-based SL, max 5% account loss (with 10% capital, 10x lev)
#  - TP = 1R / 2R / 3R
#  - Safe for Koyeb free plan (hourly run)
#  - 100-coin ready via WATCHLIST env (comma list)
#  - Fully compatible with Worker & Supabase contracts
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
OHLC_LIMIT_4H   = int(os.getenv("OHLC_LIMIT_4H", "300"))
OHLC_LIMIT_1H   = int(os.getenv("OHLC_LIMIT_1H", "400"))

PRICE_EXCHANGE  = os.getenv("PRICE_EXCHANGE", "mexc").lower()

WATCHLIST = [
    s.strip()
    for s in os.getenv("WATCHLIST",
        # default list (you can override in Koyeb env)
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,UNIUSDT,AAVEUSDT,"
        "ENAUSDT,TAOUSDT,PENDLEUSDT,VIRTUALUSDT,HBARUSDT,BCHUSDT,AVAXUSDT,"
        "LTCUSDT,DOTUSDT,SUIUSDT,NEARUSDT,OPUSDT,ARBUSDT,APTUSDT,ATOMUSDT,"
        "ALGOUSDT,INJUSDT,MATICUSDT,FTMUSDT,GALAUSDT,XLMUSDT,HYPEUSDT"
    ).split(",")
    if s.strip()
]


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


# ---------- Indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
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
def detect_swings(series: pd.Series, look: int = 3):
    """Return list of (idx, price, type) where type in {'high','low'}."""
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


def cluster_levels(swings, level_type: str, tolerance_pct: float = 0.4, min_touches: int = 3):
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
    """Bernard EMA stack trend."""
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


# ---------- BTC Regime Filter ----------
def btc_regime(ex):
    """
    Use BTC/USDT as global regime:
    - risk_on: BTC in uptrend on 4H, MACD up, RSI>50
    - risk_off: BTC in downtrend on 4H, MACD down, RSI<50
    - neutral: otherwise
    """
    try:
        df4 = fetch_tf(ex, "BTCUSDT", "4h", OHLC_LIMIT_4H)
        df1 = fetch_tf(ex, "BTCUSDT", "1h", OHLC_LIMIT_1H)

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
            return "risk_on"
        if trend4 == "down" and trend1 in ("down", "chop") and macd1_slope == "down" and rsi1 < 50:
            return "risk_off"
        return "neutral"
    except Exception as e:
        log(f"[BTC_REGIME] error: {e}")
        return "neutral"


# ---------- Candlestick helpers ----------
def bullish_retest_candle(row, level: float, tolerance_pct: float = 0.4):
    """
    Retest candle for LONG:
    - lower wick tags level (low <= level*(1+tol%))
    - close > open (bullish)
    - close above level
    """
    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]

    if l <= level * (1 + tolerance_pct/100) and h >= level * (1 - tolerance_pct/100):
        if c > o and c > level:
            return True
    return False


def bearish_retest_candle(row, level: float, tolerance_pct: float = 0.4):
    """
    Retest candle for SHORT:
    - upper wick tags level (high >= level*(1-tol%))
    - close < open (bearish)
    - close below level
    """
    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]

    if h >= level * (1 - tolerance_pct/100) and l <= level * (1 + tolerance_pct/100):
        if c < o and c < level:
            return True
    return False


# ---------- Tier Logic ----------

MAX_PRICE_RISK_PCT = 5.0  # 5% account loss cap with 10% capital & 10x lev


def tier1_break_retest(symbol, d4, d1, swings4, trend4, trend1, regime):
    """
    T1: Breakout + Retest at strong S/R (Bernard core).
    Uses 4H S/R, 1H retest candle.
    """
    # Find S/R levels on 4H
    highs = cluster_levels(swings4, "high", tolerance_pct=0.4, min_touches=4)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=0.4, min_touches=4)

    c1 = float(d1.close.iloc[-1])

    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    last = d1.iloc[-1]

    # --- LONG: breakout above resistance, pullback, bullish retest candle
    # Only if BTC not in risk_off
    if regime != "risk_off" and trend4 == "up":
        if res_level:
            # need price recently broke above res_level
            prev_close = float(d1.close.iloc[-2])
            if prev_close <= res_level and c1 > res_level * 1.001:
                # current candle retest?
                if bullish_retest_candle(last, res_level):
                    # Entry band slightly above level to avoid noise
                    entry_low = max(res_level, last["open"]) * 1.000
                    entry_high = last["close"] * 1.0015
                    entry_mid = (entry_low + entry_high) / 2.0

                    # SL = structure low (candle low a bit below)
                    sl_price = min(last["low"], res_level * 0.997)
                    price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100

                    if price_risk_pct <= MAX_PRICE_RISK_PCT:
                        return {
                            "side": "LONG",
                            "entry_low": entry_low,
                            "entry_high": entry_high,
                            "sl": sl_price,
                            "note": f"T1_BREAK_RETEST · {res_touch}x resistance @ {res_level:.4f}"
                        }

    # --- SHORT: breakdown below support, pullback, bearish retest candle
    if trend4 == "down":
        if sup_level:
            prev_close = float(d1.close.iloc[-2])
            if prev_close >= sup_level and c1 < sup_level * 0.999:
                if bearish_retest_candle(last, sup_level):
                    entry_high = min(sup_level, last["open"]) * 1.000
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
                            "note": f"T1_BREAK_RETEST · {sup_touch}x support @ {sup_level:.4f}"
                        }

    return None


def tier2_sr_bounce(symbol, d4, d1, swings4, trend4, trend1, regime):
    """
    T2: S/R or trendline-style bounce with candle confirmation.
    Slightly looser than T1 but still structure-first.
    """
    highs = cluster_levels(swings4, "high", tolerance_pct=0.5, min_touches=3)
    lows  = cluster_levels(swings4, "low",  tolerance_pct=0.5, min_touches=3)

    c1 = float(d1.close.iloc[-1])
    last = d1.iloc[-1]

    res_level, res_touch = nearest_level(highs, c1, "above")
    sup_level, sup_touch = nearest_level(lows,  c1, "below")

    # LONG: bounce off support in uptrend
    if regime != "risk_off" and trend4 == "up":
        if sup_level and abs(last["low"] - sup_level) / sup_level * 100 <= 0.5:
            # bullish bounce
            if last["close"] > last["open"] and last["close"] > sup_level:
                entry_low = max(sup_level, last["open"])
                entry_high = last["close"] * 1.001
                entry_mid = (entry_low + entry_high) / 2.0
                sl_price = min(last["low"], sup_level * 0.997)
                price_risk_pct = abs(entry_mid - sl_price) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    return {
                        "side": "LONG",
                        "entry_low": entry_low,
                        "entry_high": entry_high,
                        "sl": sl_price,
                        "note": f"T2_SR_BOUNCE · {sup_touch}x support @ {sup_level:.4f}"
                    }

    # SHORT: bounce down from resistance in downtrend
    if trend4 == "down":
        if res_level and abs(last["high"] - res_level) / res_level * 100 <= 0.5:
            if last["close"] < last["open"] and last["close"] < res_level:
                entry_high = min(res_level, last["open"])
                entry_low = last["close"] * 0.999
                entry_mid = (entry_low + entry_high) / 2.0
                sl_price = max(last["high"], res_level * 1.003)
                price_risk_pct = abs(sl_price - entry_mid) / entry_mid * 100
                if price_risk_pct <= MAX_PRICE_RISK_PCT:
                    return {
                        "side": "SHORT",
                        "entry_low": entry_low,
                        "entry_high": entry_high,
                        "sl": sl_price,
                        "note": f"T2_SR_BOUNCE · {res_touch}x resistance @ {res_level:.4f}"
                    }

    return None


def tier3_pattern(c, df1, swings):
    """
    Tier 3 — Bernard-style pattern trades
    - Only trade patterns when MOMENTUM agrees
    - Reject "naked" double tops/bottoms that fight MACD / RSI
    """

    closes = df1["close"]
    last   = float(closes.iloc[-1])

    # Need enough swings to talk about a pattern
    if len(swings) < 3:
        return (None, None, None)

    # --- 1H momentum filters (Bernard-style) ---
    # We already have df1.rsi from earlier in analyze_symbol
    try:
        r = float(df1["rsi"].iloc[-1])
    except Exception:
        # if for some reason rsi missing, skip Tier 3
        return (None, None, None)

    macd_line, macd_sig, _ = macd(df1["close"])
    macd1 = macd_slope(macd_line)

    # we only care about the last few swings
    idx = [p[0] for p in swings[-4:]]
    if len(idx) < 3:
        return (None, None, None)

    a, b, c_i = idx[-3], idx[-2], idx[-1]
    pa, pb    = float(closes.iloc[a]), float(closes.iloc[b])

    # Swings must be close in price (true "double" structure)
    # allow up to ~0.4% difference
    if abs(pa - pb) / pa * 100.0 > 0.4:
        return (None, None, None)

    # ---------- DOUBLE BOTTOM (LONG) ----------
    # Conditions:
    #  - MACD slope up (momentum building up)
    #  - RSI not oversold breakdown (>= 44) and not overheated (<= 68)
    #  - Latest price above the swing lows (bounce confirmed)
    if (
        macd1 == "up"
        and 44.0 <= r <= 68.0
        and last > pb
    ):
        lo = min(pa, pb) * 0.995   # SL zone ~0.5% below lows
        hi = last * 0.999          # entry just under current price
        note = f"T3_PATTERN · Double Bottom near {pb:.4f}"
        return ("LONG", (lo, hi), note)

    # ---------- DOUBLE TOP (SHORT) ----------
    # Conditions:
    #  - MACD slope down (momentum turning down)
    #  - RSI not exhausted down (>= 32) and not too strong up (<= 56)
    #    -> this is where FLM should be REJECTED (RSI = 60)
    #  - Latest price below the swing highs (roll-over confirmed)
    if (
        macd1 == "down"
        and 32.0 <= r <= 56.0
        and last < pb
    ):
        hi  = max(pa, pb) * 1.005  # SL zone ~0.5% above highs
        lo  = last * 1.001         # entry just below current price
        note = f"T3_PATTERN · Double Top near {pb:.4f}"
        return ("SHORT", (lo, hi), note)

    # If momentum does not agree, NO Tier-3 trade
    return (None, None, None)


# ---------- Symbol Analysis ----------
def analyze_symbol(ex, symbol: str, regime: str):
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

    # Swings for S/R & pattern
    swings4 = detect_swings(d4.close, look=3)
    swings1 = detect_swings(d1.close, look=3)

    raw_row = {
        "symbol": _mk(symbol),
        "price": last_price,
        "trend4": trend4,
        "trend1": trend1,
        "regime": regime,
        "macd4": macd4_slope,
        "macd1": macd1_slope,
        "note": "no setup"
    }

    signal = None

    # ---- Tier 1: Break + Retest (highest priority)
    sig_t1 = tier1_break_retest(symbol, d4, d1, swings4, trend4, trend1, regime)
    if sig_t1:
        signal = sig_t1
        raw_row["note"] = sig_t1["note"]

    # ---- Tier 2: SR / Trendline Bounce
    if signal is None:
        sig_t2 = tier2_sr_bounce(symbol, d4, d1, swings4, trend4, trend1, regime)
        if sig_t2:
            signal = sig_t2
            raw_row["note"] = sig_t2["note"]

    # ---- Tier 3: Double Top / Bottom pattern
    if signal is None:
        sig_t3 = tier3_pattern(symbol, d1, swings1, trend4, regime)
        if sig_t3:
            signal = sig_t3
            raw_row["note"] = sig_t3["note"]

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
            # invalid
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
            # optional telemetry for Supabase (can be null)
            "macd_slope": macd1_slope,
            "rsi": fnum(d1["rsi"].iloc[-1]),
            "atr1h_pct": fnum(d1["atr"].iloc[-1] / last_price * 100.0 if d1["atr"].iloc[-1] > 0 else None),
        }

        raw_row["side"] = side
        raw_row["signal"] = "TRIGGER"
        raw_row["sl"] = sl
        raw_row["tp1"] = tp1
        raw_row["tp2"] = tp2
        raw_row["tp3"] = tp3

        return signal_dict, raw_row

    return None, raw_row


# ---------- Worker Push (unchanged contract) ----------
def push_signals(signals):
    if not ALERT_WEBHOOK_URL or not signals:
        return
    url = ALERT_WEBHOOK_URL
    params = {"t": PASS_KEY}
    payload = {
        "summary": f"{len(signals)} signal(s) — JIM v6.1 Bernard",
        "signals": signals
    }
    try:
        with httpx.Client(timeout=10) as cli:
            r = cli.post(url, params=params, json=payload)
            print(f"[PUSH] {url} -> {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[PUSH ERROR] {e}")


# ---------- Tick Once ----------
def tick_once():
    try:
        ex = _get_exchange(PRICE_EXCHANGE)
        ex.load_markets()
    except Exception as e:
        return {"engine": "v6.1", "error": str(e)}

    regime = btc_regime(ex)

    signals = []
    raw = []

    for sym in WATCHLIST:
        try:
            sig, info = analyze_symbol(ex, sym, regime)
            raw.append(info)
            if sig:
                # If BTC regime is risk_off, block NEW longs (extra safety)
                if regime == "risk_off" and sig["side"] == "LONG":
                    info["note"] += " · blocked_by_risk_off"
                else:
                    signals.append(sig)
        except Exception as e:
            raw.append({"symbol": _mk(sym), "error": str(e)})

    if signals:
        push_signals(signals)

    return {
        "engine": "v6.1",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": raw,
        "note": dual_ts()
    }


# ---------- Timestamp helper ----------
def dual_ts():
    utc = datetime.now(timezone.utc)
    kl = utc.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
    return f"{utc:%Y-%m-%d %H:%M:%S UTC} | {kl:%Y-%m-%d %H:%M:%S KL}"
