import os, time, json, math, pathlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv
import urllib.request

# ----------- Setup ----------
ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

load_dotenv()
WORKER_URL      = os.getenv("WORKER_URL", "").rstrip("/")
PASS_KEY        = os.getenv("PASS_KEY", "")
WATCHLIST       = [s.strip() for s in os.getenv("WATCHLIST","BTCUSDT,ETHUSDT").split(",") if s.strip()]
ACCOUNT_BAL     = float(os.getenv("ACCOUNT_BALANCE","200") or 200)
LEVERAGE        = int(os.getenv("LEVERAGE","10") or 10)
RETEST_MIN_PCT  = float(os.getenv("RETEST_BAND_MIN","0.05"))
RETEST_MAX_PCT  = float(os.getenv("RETEST_BAND_MAX","0.30"))

# Binance for DATA
binance = ccxt.binance({"enableRateLimit": True})

# ----------- Helpers ----------
def ohlcv_df(klines):
    cols = ["ts","open","high","low","close","vol"]
    df = pd.DataFrame(klines, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    for c in ["open","high","low","close","vol"]:
        df[c] = df[c].astype(float)
    return df

def fetch_tf(symbol, tf, limit=400):
    s = symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"
    kl = binance.fetch_ohlcv(s, timeframe=tf, limit=limit)
    return ohlcv_df(kl)

def ema(series, n): return series.ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal    = ema(macd_line, sig)
    hist      = macd_line - signal
    return macd_line, signal, hist

def rsi(close, n=14):
    delta = close.diff()
    up = np.where(delta>0, delta, 0.0)
    dn = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(n).mean()
    roll_dn = pd.Series(dn, index=close.index).rolling(n).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    out = 100 - (100/(1+rs))
    return out.bfill().ffill().fillna(50)

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def pct(a,b):
    return abs(b-a)/a*100.0 if a else 999

def pct_prox(spot, level):
    try:
        return abs(float(spot) - float(level)) / max(1e-9, float(level)) * 100.0
    except:
        return 999.0

def donchian_levels(df, n=55):
    cap  = df["high"].rolling(n).max().iloc[-2]
    base = df["low"].rolling(n).min().iloc[-2]
    pivot = (cap + base)/2
    return cap, pivot, base

def macd_slope(line):
    if len(line) < 2: return "flat"
    return "up" if line.iloc[-1] > line.iloc[-2] else ("down" if line.iloc[-1] < line.iloc[-2] else "flat")

def build_retest_band(level, side, min_pct, max_pct):
    if side=="LONG":
        low  = level * (1 - max_pct/100.0)
        high = level * (1 - min_pct/100.0)
    else:
        low  = level * (1 + min_pct/100.0)
        high = level * (1 + max_pct/100.0)
    a,b = sorted([low,high])
    return a,b

def post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8","ignore")
        return json.loads(raw)

def log_signal(row):
    path = DATA / "signals.csv"
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=header)

# ----------- Core Scan ----------
def scan_symbol(sym):
    d4h = fetch_tf(sym, "4h")
    d1h = fetch_tf(sym, "1h")
    d15 = fetch_tf(sym, "15m")

    d4h["ema50"] = ema(d4h.close, 50)
    d4h["ema200"]= ema(d4h.close, 200)
    d4h["macd"], d4h["macdsig"], d4h["macdh"] = macd(d4h.close)
    trend4h      = "bull" if d4h.ema50.iloc[-1] > d4h.ema200.iloc[-1] else ("bear" if d4h.ema50.iloc[-1] < d4h.ema200.iloc[-1] else "flat")
    macd4_slope  = macd_slope(d4h["macd"])

    d1h["macd"], d1h["macdsig"], d1h["macdh"] = macd(d1h.close)
    d1h["rsi"]   = rsi(d1h.close,14)
    atr1h_abs    = atr(d1h,14).iloc[-1]
    atr1h_pct    = (atr1h_abs / d1h.close.iloc[-1]) * 100.0

    d15["macd"], d15["macdsig"], d15["macdh"] = macd(d15.close)
    d15["rsi"]   = rsi(d15.close,14)

    cap, pivot, base = donchian_levels(d4h, 55)
    spot = d1h.close.iloc[-1]

    allow_long  = (trend4h=="bull") or (trend4h=="flat" and macd4_slope=="up")
    allow_short = (trend4h=="bear") or (trend4h=="flat" and macd4_slope=="down")

    close1 = d1h.close.iloc[-1]
    prev1  = d1h.close.iloc[-2]
    macd1_slope = macd_slope(d1h["macd"])
    rsi1        = d1h["rsi"].iloc[-1]

    macd15_slope = macd_slope(d15["macd"])
    rsi15        = d15["rsi"].iloc[-1]

    tickets = []

    if allow_long and close1 > cap and prev1 <= cap:
        band_low, band_high = build_retest_band(cap, "LONG", RETEST_MIN_PCT, RETEST_MAX_PCT)
        entry_mid = (band_low + band_high)/2
        sl = band_low * (1 - 0.70/100.0)
        tp1 = cap * 1.011; tp2 = cap * 1.027; tp3 = cap * 1.050
        tickets.append({
            "symbol": sym, "side": "LONG", "sniper": True,
            "level": round(cap,2),
            "retest_low": round(band_low,2), "retest_high": round(band_high,2),
            "entry_mid": round(entry_mid,2),
            "sl": round(sl,2),
            "tp1": round(tp1,2), "tp2": round(tp2,2), "tp3": round(tp3,2),
            "macd_slope": macd1_slope, "rsi": round(float(rsi1),2),
            "body_close_ok": True, "retest_hold": True, "wick_trap": False,
            "body_close_ok_15m": True, "retest_hold_15m": True, "macd_slope_15m": macd15_slope,
            "atr1h_pct": round(float(atr1h_pct),2),
            "spot": round(float(spot),2), "anchor_ref": round(float(cap),2),
            "account_balance": ACCOUNT_BAL, "leverage": LEVERAGE
        })

    if allow_short and close1 < base and prev1 >= base:
        band_low, band_high = build_retest_band(base, "SHORT", RETEST_MIN_PCT, RETEST_MAX_PCT)
        entry_mid = (band_low + band_high)/2
        sl = band_high * (1 + 0.70/100.0)
        tp1 = base * 0.989; tp2 = base * 0.973; tp3 = base * 0.950
        tickets.append({
            "symbol": sym, "side": "SHORT", "sniper": True,
            "level": round(base,2),
            "retest_low": round(band_low,2), "retest_high": round(band_high,2),
            "entry_mid": round(entry_mid,2),
            "sl": round(sl,2),
            "tp1": round(tp1,2), "tp2": round(tp2,2), "tp3": round(tp3,2),
            "macd_slope": macd1_slope, "rsi": round(float(rsi1),2),
            "body_close_ok": True, "retest_hold": True, "wick_trap": False,
            "body_close_ok_15m": True, "retest_hold_15m": True, "macd_slope_15m": macd15_slope,
            "atr1h_pct": round(float(atr1h_pct),2),
            "spot": round(float(spot),2), "anchor_ref": round(float(base),2),
            "account_balance": ACCOUNT_BAL, "leverage": LEVERAGE
        })

    return {
        "symbol": sym,"trend4h": trend4h,"macd4_slope": macd4_slope,
        "cap": cap, "base": base, "spot": spot,
        "atr1h_pct": float(atr1h_pct),"tickets": tickets
    }

def main_once():
    all_rows = []
    sent = 0
    radar = []

    for sym in WATCHLIST:
        try:
            res = scan_symbol(sym)
        except Exception as e:
            print(f"[{sym}] scan error:", e)
            continue

        row={
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": res["symbol"],"trend4h": res["trend4h"],
            "cap": round(res["cap"],6),"base": round(res["base"],6),
            "spot": round(res["spot"],6),"atr1h_pct": round(res["atr1h_pct"],2),
            "tickets": len(res["tickets"])
        }
        all_rows.append(row)
        print(f"[{sym}] 4H:{res['trend4h']} cap={res['cap']:.4f} base={res['base']:.4f} spot={res['spot']:.4f} tickets={len(res['tickets'])}")

        dist_cap  = pct_prox(res["spot"], res["cap"])
        dist_base = pct_prox(res["spot"], res["base"])
        radar.append({
            "symbol": sym, "trend": res["trend4h"],
            "spot": res["spot"],"cap": res["cap"],"base": res["base"],
            "to_cap": round(dist_cap,2),"to_base": round(dist_base,2),
            "nearest": round(min(dist_cap,dist_base),2)
        })

        for t in res["tickets"]:
            try:
                url = f"{WORKER_URL}?t={PASS_KEY}"
                resp = post_json(url,t)
                sent+=1
                row2=row.copy()
                row2.update({
                    "side": t["side"],"entry_mid": t["entry_mid"],
                    "retest_low": t["retest_low"],"retest_high": t["retest_high"],
                    "sl": t["sl"],"tp1": t["tp1"],"tp2": t["tp2"],"tp3": t["tp3"],
                    "worker_ok": resp.get("ok"),"worker_state": resp.get("state"),
                    "worker_note": resp.get("note")
                })
                log_signal(row2)
                print("  -> Worker:",resp.get("state"),resp.get("note",""))
            except Exception as e:
                print("  -> Worker send error:", e)

    if radar:
        top = sorted(radar, key=lambda x: x["nearest"])[:5]
        print("\n=== SNIPER RADAR — Top 5 Nearest Levels ===")
        for r in top:
            side="LONG @cap" if r["to_cap"] < r["to_base"] else "SHORT @base"
            print(f"{r['symbol']:10s} {r['trend']:<5s} | Spot {r['spot']:.4f} | Cap {r['cap']:.4f} ({r['to_cap']}%) | Base {r['base']:.4f} ({r['to_base']}%) | Target: {side}")

    if all_rows:
        log_signal({"ts": datetime.now(timezone.utc).isoformat(),"summary_rows": len(all_rows),"sent": sent})

if __name__=="__main__":
    if not WORKER_URL or not PASS_KEY:
        print("Set WORKER_URL and PASS_KEY in .env"); raise SystemExit(1)
    main_once()
