from typing import List

# Explicitly define which columns are allowed
NUMERIC_FEATURES: List[str] = [
    "tenure",
    "monthlycharges",
    "totalcharges"
]

CATEGORICAL_FEATURES: List[str] = [
    "contract",
    "paymentmethod",
    "internetservice"
]

TARGET = "churn"
