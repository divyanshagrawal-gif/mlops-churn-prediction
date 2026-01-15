import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("monitoring/predictions.log")

def log_prediction(input_data: dict, prediction: float):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction
    }

    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
