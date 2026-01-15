import pandas as pd
from features.schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from typing import Optional, List

def build_features(df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()

    # Keep only allowed columns
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df[cols]

    # Replace empty strings and whitespace with NaN in numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            # Replace empty strings and whitespace-only strings with NaN
            df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("unknown")

    # Encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    # Align columns to match expected features (for prediction)
    if expected_columns is not None:
        # Add missing columns with 0 values
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns to match expected order and remove any extra columns
        df = df[expected_columns]
    
    return df