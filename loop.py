# loop.py — runs the JIM engine forever
import os, time, traceback
from engine import main_once

INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "15") or 15)

print(f"[JIM] loop start — scan every {INTERVAL_MIN} min")

while True:
    try:
        main_once()
    except Exception as e:
        print("[JIM] engine error:", e)
        traceback.print_exc()
    time.sleep(INTERVAL_MIN * 60)
