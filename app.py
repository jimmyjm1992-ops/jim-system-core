# app.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import os

# tick_once is optional; if engine import fails we still serve /health
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
    wl = os.getenv("WATCHLIST","").split(",") if os.getenv("WATCHLIST") else []
    return {"ok": True, "engine": "running", "watchlist": [s.strip() for s in wl if s.strip()]}

@app.get("/tick")
def tick():
    result = tick_once()
    if result is None:
        result = {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once returned None"}
    return {"ok": True, "result": result}
