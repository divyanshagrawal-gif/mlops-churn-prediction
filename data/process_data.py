import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/churn.csv")
PROCESSED_PATH = Path("data/processed/train.csv")

def main():
    df = pd.read_csv(RAW_PATH)

    # Minimal cleaning (example)
    df.columns = df.columns.str.lower()
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Processed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    main()
