# app.py
import os, time, threading
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Your engine (import your real logic) ---
from engine import tick_once  # implement in engine.py

app = FastAPI()

class PingOut(BaseModel):
    ok: bool
    msg: str

@app.get("/health", response_model=PingOut)
def health():
    return {"ok": True, "msg": "alive"}

@app.get("/tick")
def tick():
    """Manually trigger one engine iteration"""
    try:
        res = tick_once()
        return {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def loop_runner():
    """Background loop: calls engine every N seconds."""
    period_s = int(os.getenv("LOOP_PERIOD_S", "60"))
    while True:
        try:
            tick_once()
        except Exception as e:
            # keep going even if one iteration fails
            print("loop error:", e, flush=True)
        time.sleep(period_s)

# 🔹 new: start the background loop when uvicorn starts (Koyeb path)
@app.on_event("startup")
def _start_bg_loop():
    threading.Thread(target=loop_runner, daemon=True).start()

if __name__ == "__main__":
    # local dev path (python app.py)
    threading.Thread(target=loop_runner, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
