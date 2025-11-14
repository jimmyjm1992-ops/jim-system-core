# engine_loop.py — JIM v6 hourly loop (Bernard-style timing)
# - Runs FastAPI app (app.py) on port 8000
# - Every hour: waits for 1H candle close (KL time) then calls tick_once()
# - Safe for Koyeb free plan (light CPU)

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from engine import tick_once, dual_ts

KL = ZoneInfo("Asia/Kuala_Lumpur")

def run_loop():
    print("[ENGINE_LOOP] JIM v6 loop started.")
    print(f"[ENGINE_LOOP] Timezone: {KL.key} (KL)")
    scanned_this_hour = False

    while True:
        now_utc = datetime.now(timezone.utc)
        now_kl = now_utc.astimezone(KL)

        # ---- Candle timing logic ----
        # 1H candles close exactly at HH:00 KL.
        # We wait until HH:00:20 (20s buffer) to be sure the candle is fully closed
        # and exchange OHLC is updated, then run tick_once() once per hour.
        if (
            now_kl.minute == 0
            and now_kl.second >= 20
            and not scanned_this_hour
        ):
            ts = dual_ts()
            print(f"\n[ENGINE_LOOP] === Hourly scan at {ts} ===")
            try:
                result = tick_once()
                scanned = result.get("scanned", "?") if isinstance(result, dict) else "?"
                sigs = len(result.get("signals", [])) if isinstance(result, dict) else "?"
                note = result.get("note", "") if isinstance(result, dict) else ""
                print(f"[ENGINE_LOOP] Scanned: {scanned}, Signals: {sigs}, Note: {note}")
            except Exception as e:
                print(f"[ENGINE_LOOP] ERROR during tick_once: {e}")
            scanned_this_hour = True

        # Reset flag after we move away from minute 00,
        # so next hour can scan again
        if now_kl.minute != 0:
            scanned_this_hour = False

        # Sleep a bit to avoid burning CPU (Koyeb free-friendly)
        time.sleep(0.5)


if __name__ == "__main__":
    run_loop()
