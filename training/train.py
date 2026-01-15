import sys
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features.builder import build_features


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ... existing imports ...

def train_model(config: dict):
    df = pd.read_csv(config["data"]["input_path"])

    y = df[config["data"]["target"]]
    X = df.drop(columns=[config["data"]["target"]])

    X = build_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    model = LogisticRegression(**config["model"]["params"])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc, X.columns.tolist()  # Return feature names


def main(config_path: str):
    config = load_config(config_path)

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run():
        # Log config params
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_params(config["model"]["params"])
        mlflow.log_param("test_size", config["training"]["test_size"])

        model, acc, feature_names = train_model(config)  # Get feature names

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Save model locally
        joblib.dump(model, "models/model.joblib")
        
        # Save feature names
        joblib.dump(feature_names, "models/feature_names.joblib")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ChurnModel"
        )

        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
