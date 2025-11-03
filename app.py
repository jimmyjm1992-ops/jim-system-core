# app.py — FastAPI surface for health + manual tick

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from engine import tick_once, WATCHLIST  # import from engine

app = FastAPI(title="JIM System Core", version="1.0.0")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True, "engine": "running", "watchlist": WATCHLIST}

@app.get("/tick")
def tick():
    result = tick_once()
    if result is None:
        result = {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once returned None"}
    return {"ok": True, "result": result}
