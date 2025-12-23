from fastapi import FastAPI
from pydantic import BaseModel

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
    return {"message": "Fake News Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewsRequest):
    prob = predict_news(request.text)

    label = "Fake" if prob >= 0.5 else "Real"

    return {
        "fake_probability": round(prob, 4),
        "label": label
    }
