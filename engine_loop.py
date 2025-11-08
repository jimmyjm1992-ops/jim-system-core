# engine_loop.py — background loop w/ dual timestamps (UTC | KL)
import os, time, traceback
from engine import tick_once, dual_ts

# keep the engine warm; both paths = 300s
SLEEP_OK   = int(os.getenv("LOOP_SLEEP_OK", "300"))   # 5 min
SLEEP_IDLE = int(os.getenv("LOOP_SLEEP_IDLE", "300")) # 5 min

if __name__ == "__main__":
    print("[loop] started")
    while True:
        try:
            res = tick_once()
            n = len(res.get("signals", []))
            print(f"[{dual_ts()}] [loop] tick scanned={res.get('scanned')} signals={n}")
            time.sleep(SLEEP_OK if n > 0 else SLEEP_IDLE)
        except KeyboardInterrupt:
            print("[loop] stopped by user")
            break
        except Exception as e:
            print("[loop] error:", e)
            traceback.print_exc()
            time.sleep(30)
