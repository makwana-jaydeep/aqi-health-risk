import argparse
import json
import logging
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "no2",
    "temperature", "humidity", "wind_speed",
    "age_group_enc", "has_asthma", "has_heart_condition", "planned_activity_enc",
]


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    params = load_params()
    args = parse_args()

    model_params = params["model"]
    n_estimators = args.n_estimators or model_params["n_estimators"]
    max_depth = args.max_depth or model_params["max_depth"]
    learning_rate = args.learning_rate or model_params["learning_rate"]
    random_state = args.random_state or model_params["random_state"]
    test_size = model_params["test_size"]

    processed_path = os.path.join(params["data"]["processed_path"], "aqi_processed.csv")
    model_path = "data/model.pkl"
    metrics_path = "data/metrics.json"

    logger.info("Loading processed data from %s", processed_path)
    df = pd.read_csv(processed_path)

    X = df[FEATURE_COLUMNS]
    y = df["risk_tier"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train size=%d Test size=%d", len(X_train), len(X_test))

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("aqi-risk-classification")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "test_size": test_size,
            "train_samples": len(X_train),
        })

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )),
        ])

        logger.info("Training model n_estimators=%d max_depth=%d lr=%s", n_estimators, max_depth, learning_rate)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))

        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        })

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=model_params["name"],
        )

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "run_id": run.info.run_id,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Metrics: accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f", accuracy, f1_macro, f1_weighted)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    version_file = model_path.replace(".pkl", "_version.txt")
    with open(version_file, "w") as f:
        f.write(run.info.run_id[:8])

    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
