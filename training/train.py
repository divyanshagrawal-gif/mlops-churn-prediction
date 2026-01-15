import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features.build_features import build_features

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)

    df = pd.read_csv(config["data"]["input_path"])
    y = df[config["data"]["target"]]
    X = df.drop(columns=[config["data"]["target"]])

    X = build_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    model = LogisticRegression(**config["model"]["params"])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    joblib.dump(model, "models/model.joblib")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
