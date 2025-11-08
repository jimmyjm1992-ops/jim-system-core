from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import os

# We try to import tick_once; if it fails we still keep /health alive.
try:
    from engine import tick_once
except Exception:
    def tick_once():
        return {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once() unavailable"}

app = FastAPI(title="JIM System Core", version="0.1.0")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    wl = os.getenv("WATCHLIST", "")
    symbols = [s.strip() for s in wl.split(",") if s.strip()] if wl else []
    return {"ok": True, "engine": "running", "watchlist": symbols}

@app.get("/tick")
def tick():
    result = tick_once()
    if result is None:
        result = {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once() returned None"}
    return {"ok": True, "result": result}
