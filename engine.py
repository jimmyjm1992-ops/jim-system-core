# engine.py — JIM V5 framework scaffold online

import os
import ccxt

exchange = ccxt.mexc()

WATCHLIST = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
    "HYPE/USDT","XLM/USDT","SUI/USDT","LINK/USDT","HBAR/USDT",
    "BCH/USDT","AVAX/USDT","LTC/USDT","DOT/USDT","UNI/USDT",
    "AAVE/USDT","ENA/USDT","TAO/USDT","PENDLE/USDT","VIRTUAL/USDT"
]

def fetch_last(symbol):
    try:
        t = exchange.fetch_ticker(symbol)
        return t.get("last")
    except:
        return None

def tick_once():
    result = []

    for sym in WATCHLIST:
        price = fetch_last(sym)

        result.append({
            "symbol": sym,
            "price": price,
            "bias": "pending",
            "zone": "pending",
            "signal": "none",
            "note": "framework boot — indicators not applied yet"
        })

    return {
        "engine":"jim_v5_boot",
        "scanned":len(result),
        "signals":[r for r in result if r["signal"] != "none"],
        "raw": result[:5],  # show first 5 only
        "note":"ready for breakout + retest logic"
    }
