import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1y", "1-2y", "2-4y", "4-6y"]
    )

    df = pd.get_dummies(df, drop_first=True)
    return df
