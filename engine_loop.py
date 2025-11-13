# engine_loop.py
# JIM Framework v6 — hourly scan loop (Malaysia time)
# Runs tick_once() once per hour, just after the 1H candle close.

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from engine import tick_once, dual_ts  # engine v6


KL_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def main():
    print("[ENGINE_LOOP] JIM v6 loop started.")
    print("[ENGINE_LOOP] Timezone: Asia/Kuala_Lumpur (KL)")
    last_run_hour = None

    while True:
        now_kl = datetime.now(KL_TZ)

        # Run once per hour when minute == 59, and only once for that hour
        if now_kl.minute == 59 and last_run_hour != now_kl.hour:
            ts = dual_ts()
            print(f"\n[ENGINE_LOOP] === Hourly scan at {ts} ===")

            try:
                result = tick_once()
            except Exception as e:
                print(f"[ENGINE_LOOP] tick_once() error: {e}")
                result = None

            if isinstance(result, dict):
                scanned = result.get("scanned")
                num_signals = len(result.get("signals", []))
                note = result.get("note", "")
                print(f"[ENGINE_LOOP] Scanned: {scanned}, Signals: {num_signals}, Note: {note}")
            else:
                print("[ENGINE_LOOP] tick_once() returned non-dict result")

            # Mark this hour as done so we don't double-run
            last_run_hour = now_kl.hour

            # Sleep a bit longer to move away from minute 59 window
            time.sleep(65)
        else:
            # Light sleep, keeps CPU low on Koyeb free plan
            time.sleep(15)


if __name__ == "__main__":
    main()
