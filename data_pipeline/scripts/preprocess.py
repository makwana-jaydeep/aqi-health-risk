import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_pipeline.scripts.feature_engineering import engineer_features
from ml.drift_detection import compute_baseline_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    raw_path = os.path.join(params["data"]["raw_path"], "aqi_dataset.csv")
    processed_dir = params["data"]["processed_path"]
    processed_path = os.path.join(processed_dir, "aqi_processed.csv")
    baseline_path = params["data"]["baseline_path"]

    os.makedirs(processed_dir, exist_ok=True)

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Raw data shape: %s", df.shape)

    df = _clean_data(df)
    df = engineer_features(df)

    df.to_csv(processed_path, index=False)
    logger.info("Processed data saved to %s shape=%s", processed_path, df.shape)

    compute_baseline_stats(df, baseline_path)
    logger.info("Baseline stats saved to %s", baseline_path)


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)
    df = df.dropna(subset=["aqi", "pm25", "pm10"])
    df = df[df["aqi"].between(0, 500)]
    df = df[df["pm25"] >= 0]
    df = df[df["pm10"] >= 0]
    df = df[df["humidity"].between(0, 100)]

    for col in ["aqi", "pm25", "pm10", "no2", "temperature", "humidity", "wind_speed"]:
        if col in df.columns:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q_low, upper=q_high)

    logger.info("Cleaned data: %d -> %d rows", initial_len, len(df))
    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()
