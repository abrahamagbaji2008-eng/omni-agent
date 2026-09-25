from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="OmniAgent API")


class HealthResponse(BaseModel):
    status: str


@app.get("/")
def read_root():
    return {"message": "Hello from OmniAgent backend!"}


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}
