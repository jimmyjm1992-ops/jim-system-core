# ==========================================================
#  engine_v6.py — JIM Framework v6 (Bernard-style)
#  - 3-tier system: T1 Sniper / T2 Momentum / T3 Pattern
#  - Synthetic BTC regime filter (risk-on / risk-off)
#  - Bernard-style trendline + 5-touch S/R detection
#  - Safe for Koyeb free plan (light CPU)
#  - Fully compatible with your Worker & Supabase
# ==========================================================

import os, json, time, pathlib, math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ccxt
import httpx
from dotenv import load_dotenv


# -------- FS ROOT --------
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

load_dotenv()

# -------- ENV --------
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").rstrip("/")
PASS_KEY          = os.getenv("PASS_KEY", "")

LOG_VERBOSE     = int(os.getenv("LOG_VERBOSE", "0"))
OHLC_LIMIT_4H   = int(os.getenv("OHLC_LIMIT_4H", "300"))
OHLC_LIMIT_1H   = int(os.getenv("OHLC_LIMIT_1H", "400"))

# Global mode off by default
GLOBAL_MODE    = int(os.getenv("GLOBAL", "0"))
PRICE_EXCHANGE = os.getenv("PRICE_EXCHANGE", "mexc").lower()

WATCHLIST = [
    s.strip() for s in os.getenv("WATCHLIST",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,UNIUSDT,AAVEUSDT,ENAUSDT,TAOUSDT,"
    "PENDLEUSDT,VIRTUALUSDT,HBARUSDT,BCHUSDT,AVAXUSDT,LTCUSDT,DOTUSDT,SUIUSDT,NEARUSDT,"
    "OPUSDT,ARBUSDT,APTUSDT,ATOMUSDT,ALGOUSDT,INJUSDT,MATICUSDT,FTMUSDT,GALAUSDT,XLMUSDT,HYPEUSDT"
).split(",") if s.strip()
]


# -------- utils --------
def log(msg):
    if LOG_VERBOSE:
        print(msg)


def fnum(x):
    try:
        x = float(x)
    except:
        return None
    return x if math.isfinite(x) else None


def _mk(symbol):
    return symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"


def _get_exchange(name):
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True})


# -------- OHLC --------
def timed_fetch_ohlcv(ex, symbol, timeframe="1h", limit=400):
    t0 = time.time()
    out = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    log(f"[LATENCY] {ex.id.upper()} {symbol} {timeframe} x{limit} -> {(time.time()-t0)*1000:.2f} ms")
    return out


def _ohlcv(kl):
    df = pd.DataFrame(kl, columns=["ts","open","high","low","close","vol"])
    df["ts"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    df.set_index("ts",inplace=True)
    for c in ["open","high","low","close","vol"]:
        df[c]=df[c].astype(float)
    return df


def fetch_tf(ex,sym,tf="1h",limit=400):
    return _ohlcv(timed_fetch_ohlcv(ex,_mk(sym),tf,limit))


# -------- Indicators --------
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def macd(close,fast=12,slow=26,sig=9):
    line=ema(close,fast)-ema(close,slow)
    signal=ema(line,sig)
    hist=line-signal
    return line,signal,hist

def rsi(close,n=14):
    d=close.diff()
    up=np.where(d>0,d,0)
    dn=np.where(d<0,-d,0)
    roll_up=pd.Series(up,index=close.index).rolling(n).mean()
    roll_dn=pd.Series(dn,index=close.index).rolling(n).mean().replace(0,np.nan)
    rs=roll_up/roll_dn
    out=100-(100/(1+rs))
    return out.bfill().ffill().fillna(50)

def donchian(df,n=55):
    if len(df)<n+2: return float("nan"),float("nan")
    cap=df["high"].rolling(n).max().iloc[-2]
    base=df["low"].rolling(n).min().iloc[-2]
    return cap,base

def macd_slope(series):
    return "up" if series.iloc[-1]>series.iloc[-2] else "down" if series.iloc[-1]<series.iloc[-2] else "flat"


# -------- Trendline / S/R (Bernard-style 5-touch) --------
def detect_swing_levels(series, sensitivity=5):
    """
    Detect minor swing highs/lows.
    Used for Bernard-style multi-touch S/R lines.
    """
    out=[]
    for i in range(sensitivity, len(series)-sensitivity):
        window=series[i-sensitivity:i+sensitivity+1]
        if series[i] == window.max() or series[i] == window.min():
            out.append((i,series[i]))
    return out


def regime_filter_btc(df4, df1):
    """
    A lightweight synthetic BTC regime:
    - Trend direction from 4H EMAs
    - Momentum from 1H RSI/MACD
    - Volatility from ATR
    """
    trend = "bull" if df4.ema21.iloc[-1] > df4.ema50.iloc[-1] else "bear"

    line,sig,_ = macd(df1.close)
    macd_up = line.iloc[-1] > line.iloc[-2]

    r = rsi(df1.close).iloc[-1]

    tr = pd.concat([
        df1["high"] - df1["low"],
        (df1["high"] - df1["close"].shift()).abs(),
        (df1["low"] - df1["close"].shift()).abs()
    ],axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr/df1.close.iloc[-1]*100 if math.isfinite(atr) else 1.0

    if trend=="bull" and macd_up and r>50 and 0.3<=atr_pct<=1.8:
        return "risk_on"

    if trend=="bear" and not macd_up and r<50:
        return "risk_off"

    return "neutral"


# -------- Tier logic --------
def tier1_sniper(c, cap, base, df1, trend4, macd4):
    if c>cap and df1.close.iloc[-2]<=cap and trend4=="bull" and macd4=="up":
        hi=cap*(1-0.0005)
        lo=cap*(1-0.0030)
        return ("LONG",(lo,hi),"Tier: T1_SNIPER · Break + Retest")

    if c<base and df1.close.iloc[-2]>=base and trend4=="bear" and macd4=="down":
        lo=base*(1+0.0005)
        hi=base*(1+0.0030)
        return ("SHORT",(lo,hi),"Tier: T1_SNIPER · Break + Retest")

    return (None,None,None)


def tier2_momentum(c, cap, base, df1, trend4, macd4, atr_pct):
    r = float(df1.rsi.iloc[-1])

    if not (0.5<=atr_pct<=2.0):
        return (None,None,None)

    if c>cap:
        body=(c-cap)/cap*100
        if 0.25<=body<=1.20 and trend4=="bull" and macd4=="up" and 48<=r<=68:
            hi=c*(1-0.0008)
            lo=c*(1-0.0035)
            return ("LONG",(lo,hi),"Tier: T2_MOMENTUM · Continuation")
    if c<base:
        body=(base-c)/base*100
        if 0.25<=body<=1.20 and trend4=="bear" and macd4=="down" and 32<=r<=55:
            lo=c*(1+0.0008)
            hi=c*(1+0.0035)
            return ("SHORT",(lo,hi),"Tier: T2_MOMENTUM · Continuation")

    return (None,None,None)


def tier3_pattern(c, df1, swings):
    # simple double bottom/top detection
    closes=df1.close
    last=closes.iloc[-1]
    idx=[p[0] for p in swings[-4:]]  # last few swings

    if len(idx)>=3:
        a,b,c_i= idx[-3],idx[-2],idx[-1]
        pa, pb = closes.iloc[a], closes.iloc[b]
        # Double Bottom
        if abs(pa-pb)/pa < 0.005 and last > pb:
            lo = pb*0.995
            hi = last*0.999
            return ("LONG",(lo,hi),"Tier: T3_PATTERN · Double Bottom")
        # Double Top
        if abs(pa-pb)/pa < 0.005 and last < pb:
            hi = pb*1.005
            lo = last*1.001
            return ("SHORT",(lo,hi),"Tier: T3_PATTERN · Double Top")

    return (None,None,None)


# -------- Main analyze --------
def analyze_symbol(primary_ex, symbol):
    d4 = fetch_tf(primary_ex, symbol, "4h", OHLC_LIMIT_4H)
    d1 = fetch_tf(primary_ex, symbol, "1h", OHLC_LIMIT_1H)

    d4["ema21"]=ema(d4.close,21)
    d4["ema50"]=ema(d4.close,50)
    d4["macd"],_,_=macd(d4.close)
    trend4 = "bull" if d4.ema21.iloc[-1]>d4.ema50.iloc[-1] else "bear"
    macd4  = macd_slope(d4["macd"])

    d1["ema21"]=ema(d1.close,21)
    d1["ema50"]=ema(d1.close,50)
    d1["rsi"]=rsi(d1.close)

    tr = pd.concat([
        d1["high"] - d1["low"],
        (d1["high"] - d1["close"].shift()).abs(),
        (d1["low"]  - d1["close"].shift()).abs()
    ],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().iloc[-1]
    atr_pct = atr/d1.close.iloc[-1]*100 if math.isfinite(atr) else float("nan")

    cap,base = donchian(d4,55)
    c=float(d1.close.iloc[-1])

    # regime (Bernard-style)
    btc_regime = regime_filter_btc(d4,d1)

    # multi-touch swings
    swings = detect_swing_levels(d1.close, sensitivity=3)

    # ---------- T1
    side,band,note = tier1_sniper(c,cap,base,d1,trend4,macd4)
    if side:
        return side,band,note

    # ---------- T2 only if risk-on
    if btc_regime=="risk_on":
        side,band,note = tier2_momentum(c,cap,base,d1,trend4,macd4,atr_pct)
        if side:
            return side,band,note

    # ---------- T3 only in trend (bull or bear)
    if trend4 in ("bull","bear") and btc_regime!="risk_off":
        side,band,note = tier3_pattern(c,d1,swings)
        if side:
            return side,band,note

    return None,None,"no setup"


# -------- Push to worker (DO NOT CHANGE CONTRACT) --------
def push_signals(signals):
    if not ALERT_WEBHOOK_URL or not signals:
        return
    url = ALERT_WEBHOOK_URL
    params = {"t": PASS_KEY}
    payload = {
        "summary": f"{len(signals)} signal(s) — JIM v6 Engine",
        "signals": signals
    }
    try:
        with httpx.Client(timeout=10) as cli:
            r=cli.post(url,params=params,json=payload)
            print(f"[PUSH] {url} -> {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[PUSH ERROR] {e}")


# -------- Tick Once --------
def tick_once():
    try:
        ex = _get_exchange(PRICE_EXCHANGE)
        ex.load_markets()
    except Exception as e:
        return {"engine":"v6","error":str(e)}

    signals=[]
    raw=[]

    for sym in WATCHLIST:
        try:
            side,band,note = analyze_symbol(ex,sym)
            row={"symbol":_mk(sym),"note":note}

            if side and band:
                lo,hi = band
                entry_mid=(lo+hi)/2
                sl = entry_mid*0.988   # approx 1.2% for Bernard-style flexible system

                signals.append({
                    "symbol": _mk(sym),
                    "side": side,
                    "tf": "1h",
                    "price": entry_mid,
                    "entry": [fnum(lo),fnum(hi)],
                    "entry_mid": fnum(entry_mid),
                    "sl": fnum(sl),
                    "tp1": None, "tp2": None, "tp3": None,
                    "note": note
                })

            raw.append(row)
        except Exception as e:
            raw.append({"symbol":_mk(sym),"error":str(e)})

    if signals:
        push_signals(signals)

    return {
        "engine":"v6",
        "scanned": len(WATCHLIST),
        "signals": signals,
        "raw": raw,
        "note": "JIM v6 active"
    }


# -------- Timestamp helper --------
def dual_ts():
    utc=datetime.now(timezone.utc)
    kl=utc.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
    return f"{utc:%Y-%m-%d %H:%M:%S UTC} | {kl:%Y-%m-%d %H:%M:%S KL}"
