# ---------- Base ----------
FROM python:3.10-slim

# Prevent pyc files, force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- Deps ----------
# (keep layers small; leverage wheels on slim)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------- App code ----------
COPY . .

# Koyeb health check listens on 8000 (your FastAPI)
EXPOSE 8000

# ---------- Start both processes ----------
# 1) engine_loop.py  -> your scanner/engine (background)
# 2) app.py          -> FastAPI app object named "app"
CMD ["bash", "-lc", "python engine_loop.py & uvicorn app:app --host 0.0.0.0 --port 8000"]
