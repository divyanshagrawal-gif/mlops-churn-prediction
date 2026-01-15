from fastapi import FastAPI
from serving.schemas import ChurnRequest, ChurnResponse
from serving.predictor import ChurnPredictor
from monitoring.logger import log_prediction

app = FastAPI(title="Churn Prediction API")

predictor = ChurnPredictor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    payload = request.model_dump()
    prob = predictor.predict(request.model_dump())
    log_prediction(payload, prob)
    return ChurnResponse(churn_probability=prob)
