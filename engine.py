# engine.py
import os
import ccxt

# Build the pair list: from PAIRS env or default 20
DEFAULT_PAIRS = (
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,"
    "HYPE/USDT,XLM/USDT,SUI/USDT,LINK/USDT,HBAR/USDT,"
    "BCH/USDT,AVAX/USDT,LTC/USDT,DOT/USDT,UNI/USDT,"
    "AAVE/USDT,ENA/USDT,TAO/USDT,PENDLE/USDT,VIRTUAL/USDT"
)

def _normalize_pairs(raw: str):
    items = [p.strip() for p in raw.split(",") if p.strip()]
    # Accept BTCUSDT or BTC/USDT; convert to CCXT format
    fixed = []
    for p in items:
        if "/" in p:
            fixed.append(p.upper())
        else:
            # e.g. BTCUSDT -> BTC/USDT
            if p.upper().endswith("USDT"):
                fixed.append(p[:-4].upper() + "/USDT")
            else:
                fixed.append(p.upper())
    return fixed

PAIRS = _normalize_pairs(os.getenv("PAIRS", DEFAULT_PAIRS))

# CCXT public client (no keys required for fetch_ticker)
exchange = ccxt.mexc()

def tick_once():
    """
    Minimal 'scan' — ping public ticker for each symbol to prove
    engine wiring works. Replace with your JIM rules later.
    """
    scanned = 0
    got = []
    errors = []

    for sym in PAIRS:
        try:
            t = exchange.fetch_ticker(sym)
            scanned += 1
            got.append({"symbol": sym.replace("/", ""), "last": t.get("last")})
        except Exception as e:
            # keep going even if one fails
            errors.append({"symbol": sym, "error": str(e)[:120]})

    note = "public fetch_ticker via CCXT"
    if errors:
        note += f" | {len(errors)} error(s)"

    return {
        "engine": "ok",
        "scanned": scanned,
        "pairs": [g["symbol"] for g in got],
        "note": note,
        "sample": got[:5],   # small sample so you can see prices
        "errors": errors[:3] # show a couple if any
    }
