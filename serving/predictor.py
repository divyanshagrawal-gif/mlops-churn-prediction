import joblib
import pandas as pd
from pathlib import Path

from features.builder import build_features

MODEL_PATH = Path("models/model.joblib")
FEATURE_NAMES_PATH = Path("models/feature_names.joblib")

class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        # Load expected feature names
        if FEATURE_NAMES_PATH.exists():
            self.expected_features = joblib.load(FEATURE_NAMES_PATH)
        elif hasattr(self.model, 'feature_names_in_'):
            self.expected_features = self.model.feature_names_in_.tolist()
        else:
            self.expected_features = None

    def predict(self, payload: dict) -> float:
        df = pd.DataFrame([payload])
        X = build_features(df, expected_columns=self.expected_features)
        prob = self.model.predict_proba(X)[0][1]
        return float(prob)