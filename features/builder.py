import pandas as pd
from features.schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only allowed columns
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df[cols]

    # Handle missing values
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("unknown")

    # Encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    return df
