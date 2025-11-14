# engine_loop.py
# JIM Framework v6 — hourly scan loop (Malaysia time)
# Runs tick_once() once per hour, shortly AFTER the 1H candle close.

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from engine import tick_once, dual_ts  # engine v6

KL_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def main():
    print("[ENGINE_LOOP] JIM v6 loop started.")
    print("[ENGINE_LOOP] Timezone: Asia/Kuala_Lumpur (KL)")

    last_hour = None
    scanned_this_hour = False

    while True:
        now_kl = datetime.now(KL_TZ)

        # When the hour changes, allow a new scan
        if last_hour is None or now_kl.hour != last_hour:
            last_hour = now_kl.hour
            scanned_this_hour = False

        # Run once per hour, AFTER candle close:
        # KL candle closes at HH:00:00 → we scan at HH:00:20+
        if (
            not scanned_this_hour
            and now_kl.minute == 0
            and now_kl.second >= 20
        ):
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
                print(
                    f"[ENGINE_LOOP] Scanned: {scanned}, "
                    f"Signals: {num_signals}, Note: {note}"
                )
            else:
                print("[ENGINE_LOOP] tick_once() returned non-dict result")

            # Mark as done for this hour
            scanned_this_hour = True

        # Light sleep keeps CPU low on Koyeb free plan but
        # still precise enough to catch the 00:20 window.
        time.sleep(0.5)


if __name__ == "__main__":
    main()
