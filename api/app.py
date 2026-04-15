from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from inference.predict import predict_news


# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(
    title="Fake News Detection API",
    description="Hybrid BERT + GNN + Fusion model for Fake News Detection",
    version="1.0"
)


# =====================================================
# Static files
# =====================================================
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# =====================================================
# Request schema
# =====================================================
class NewsRequest(BaseModel):
    text: str


# =====================================================
# Response schema
# =====================================================
class PredictionResponse(BaseModel):
    fake_probability: float
    label: str


# =====================================================
# Routes
# =====================================================
@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/health")
def health():
    return {"message": "Fake News Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewsRequest):
    prob = predict_news(request.text)

    label = "Fake" if prob >= 0.5 else "Real"

    return {
        "fake_probability": round(prob, 4),
        "label": label
    }
