
# ---------- Base ----------
FROM python:3.10-slim

# Prevent pyc files, force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- Deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------- App code ----------
COPY . .

# Health/API port
EXPOSE 8000

# ---------- Start both processes ----------
# 1) engine_loop.py  -> scanner (background)
# 2) app.py          -> FastAPI (serves /health, /tick)
CMD ["bash", "-lc", "python bernard_v12_engine.py & uvicorn app:app --host 0.0.0.0 --port 8000"]

