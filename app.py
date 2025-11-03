# app.py (confirm these exist)
from flask import Flask, jsonify
import threading, time

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(ok=True, msg="alive")

def run_loop():
    # import your engine here and start the background loop
    while True:
        # engine.tick()  # your periodic work
        time.sleep(60)

if __name__ == "__main__":
    threading.Thread(target=run_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8000)
