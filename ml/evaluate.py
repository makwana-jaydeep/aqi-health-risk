import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "no2",
    "temperature", "humidity", "wind_speed",
    "age_group_enc", "has_asthma", "has_heart_condition", "planned_activity_enc",
]


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    processed_path = os.path.join(params["data"]["processed_path"], "aqi_processed.csv")
    model_path = "data/model.pkl"
    eval_metrics_path = "data/eval_metrics.json"

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading evaluation data from %s", processed_path)
    df = pd.read_csv(processed_path)

    from sklearn.model_selection import train_test_split
    X = df[FEATURE_COLUMNS]
    y = df["risk_tier"]
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
        stratify=y,
    )

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=["Safe", "Caution", "Avoid Outdoors"],
        output_dict=True,
    )

    eval_metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info("Evaluation complete: accuracy=%.4f f1_macro=%.4f", accuracy, f1_macro)
    report_str = classification_report(y_test, y_pred, target_names=["Safe", "Caution", "Avoid Outdoors"])
    logger.info("Classification report:\n%s", report_str)


if __name__ == "__main__":
    main()
