# app.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse
import os

try:
    from engine import tick_once
except Exception:
    def tick_once():
        return {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once returned None"}

app = FastAPI(title="JIM System Core", version="0.1.0")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    wl = os.getenv("WATCHLIST", "").split(",") if os.getenv("WATCHLIST") else []
    return {"ok": True, "engine": "running", "watchlist": [s.strip() for s in wl if s.strip()]}

@app.get("/tick")
def tick():
    result = tick_once()
    if result is None:
        result = {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once returned None"}
    return {"ok": True, "result": result}

@app.get("/summary", response_class=HTMLResponse)
def summary():
    """HTML summary table (works fine on mobile / Opera VPN)."""
    res = tick_once()
    rows = []
    for r in res.get("raw", []):
        sym = r.get("symbol", "-")
        price = r.get("price", "-")
        bias = r.get("bias", "-")
        zone = r.get("zone", "-")
        signal = r.get("signal", "-")
        note = r.get("note", "-")
        rows.append(f"<tr><td>{sym}</td><td>{price}</td><td>{bias}</td><td>{zone}</td><td>{signal}</td><td>{note}</td></tr>")
    html = f"""
    <html>
    <head>
        <title>JIM Summary</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; background: #0d1117; color: #c9d1d9; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #30363d; padding: 8px; text-align: left; }}
            th {{ background: #161b22; }}
            tr:nth-child(even) {{ background: #1c2128; }}
            tr:hover {{ background: #21262d; }}
            h1 {{ text-align: center; }}
        </style>
    </head>
    <body>
        <h1>JIM v5.9.2 — Summary</h1>
        <p>Scanned: {res.get("scanned")} pairs | Signals: {len(res.get("signals", []))}</p>
        <table>
            <tr><th>Symbol</th><th>Price</th><th>Bias</th><th>Zone</th><th>Signal</th><th>Note</th></tr>
            {''.join(rows)}
        </table>
        <p style="text-align:center;">Last updated: {res.get("note")}</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
