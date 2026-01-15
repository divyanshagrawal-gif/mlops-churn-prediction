from pydantic import BaseModel

class ChurnRequest(BaseModel):
    tenure: int
    monthlycharges: float
    totalcharges: float
    contract: str
    paymentmethod: str
    internetservice: str


class ChurnResponse(BaseModel):
    churn_probability: float
