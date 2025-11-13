# app.py — HTTP wrapper for JIM v6 engine
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from engine import tick_once, dual_ts

app = FastAPI(title="JIM System Core — v6", version="6.0")

@app.get("/")
def root():
    return RedirectResponse(url="/health")

@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "jim_v6",
        "ts": dual_ts(),
    }

@app.post("/run")
def run():
    """
    One full scan + push to Worker (if any signals).
    Called by Koyeb cron every hour.
    """
    result = tick_once()
    return {
        "ok": True,
        "engine": result.get("engine"),
        "scanned": result.get("scanned", 0),
        "signals": len(result.get("signals", [])),
        "note": result.get("note"),
    }
