from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "engine": "bernard_v12"}

@app.get("/health")
def health():
    return {"status": "ok"}
