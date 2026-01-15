import json
import pandas as pd
from pathlib import Path

from monitoring.drift import detect_drift

TRAIN_DATA = Path("data/processed/train.csv")
LOG_FILE = Path("monitoring/predictions.log")

def main():
    train_df = pd.read_csv(TRAIN_DATA)

    records = []
    with open(LOG_FILE) as f:
        for line in f:
            records.append(json.loads(line)["input"])

    prod_df = pd.DataFrame(records)

    report = detect_drift(train_df, prod_df)

    print("DRIFT REPORT")
    for k, v in report.items():
        print(k, v)

if __name__ == "__main__":
    main()
