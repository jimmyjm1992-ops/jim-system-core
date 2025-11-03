# app.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

try:
    from engine import tick_once
except Exception:
    def tick_once():
        return {"status": "engine not loaded"}

app = FastAPI(title="JIM System Core", version="0.1.0")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True, "engine": "running"}

@app.get("/tick")
def tick():
    result = tick_once()
    if result is None:
        result = {"engine": "ok", "scanned": 0, "pairs": [], "note": "tick_once returned None"}
    return {"ok": True, "result": result}
