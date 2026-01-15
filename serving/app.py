from fastapi import FastAPI
from serving.schemas import ChurnRequest, ChurnResponse
from serving.predictor import ChurnPredictor

app = FastAPI(title="Churn Prediction API")

predictor = ChurnPredictor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    prob = predictor.predict(request.model_dump())
    return ChurnResponse(churn_probability=prob)
