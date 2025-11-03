# app.py
import os, threading, time
from flask import Flask, Response
from engine import main_once  # your existing scan+send routine

app = Flask(__name__)

def run_loop():
    # run every 15 minutes; adjust as you like
    while True:
        try:
            print("[loop] tick")
            main_once()
        except Exception as e:
            print("[loop] error:", e)
        time.sleep(15 * 60)

# start background loop
threading.Thread(target=run_loop, daemon=True).start()

@app.get("/")
def root():
    return Response("ok", status=200, mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
