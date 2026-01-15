import pandas as pd
from scipy.stats import ks_2samp

from features.schema import NUMERIC_FEATURES


def detect_drift(train_df: pd.DataFrame, prod_df: pd.DataFrame, threshold=0.05):
    drift_report = {}

    for feature in NUMERIC_FEATURES:
        # Ensure both dataframes have numeric types for this feature
        train_values = pd.to_numeric(train_df[feature], errors='coerce')
        prod_values = pd.to_numeric(prod_df[feature], errors='coerce')
        
        # Drop NaN values that might result from conversion
        train_values = train_values.dropna()
        prod_values = prod_values.dropna()
        
        if len(train_values) == 0 or len(prod_values) == 0:
            drift_report[feature] = {
                "p_value": None,
                "drift_detected": None,
                "error": "Insufficient data after type conversion"
            }
            continue
        
        stat, p_value = ks_2samp(train_values, prod_values)
        drift_report[feature] = {
            "p_value": p_value,
            "drift_detected": p_value < threshold
        }

    return drift_report