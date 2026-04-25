import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CONTINUOUS_FEATURES = ["aqi", "pm25", "pm10", "no2", "temperature", "humidity", "wind_speed"]

# to compute drift
def compute_baseline_stats(df: pd.DataFrame, output_path: str) -> None:
    baseline = {}
    for feature in CONTINUOUS_FEATURES:
        if feature not in df.columns:
            continue
        values = df[feature].dropna().tolist()
        baseline[feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "samples": values[:2000],
        }

    with open(output_path, "w") as f:
        json.dump(baseline, f)

    logger.info("Baseline stats saved to %s features=%d", output_path, len(baseline))


def detect_drift(
    baseline_path: str,
    current_data: Dict[str, List[float]],
    ks_threshold: float = 0.05,
) -> Dict[str, Any]:
    with open(baseline_path) as f:
        baseline = json.load(f)

    results = {}
    drift_detected = False

    for feature in CONTINUOUS_FEATURES:
        if feature not in baseline or feature not in current_data:
            continue
        baseline_samples = np.array(baseline[feature]["samples"])
        current_samples = np.array(current_data[feature])

        if len(current_samples) < 10:
            continue

        ks_stat, p_value = stats.ks_2samp(baseline_samples, current_samples)
        drifted = p_value < ks_threshold

        results[feature] = {
            "ks_stat": float(ks_stat),
            "p_value": float(p_value),
            "drifted": drifted,
        }

        if drifted:
            drift_detected = True
            logger.warning(
                "Drift in %s: ks_stat=%.4f p=%.4f", feature, ks_stat, p_value
            )

    return {"drift_detected": drift_detected, "features": results}


if __name__ == "__main__":
    import yaml

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    baseline_path = params["data"]["baseline_path"]
    processed_path = os.path.join(params["data"]["processed_path"], "aqi_processed.csv")

    df = pd.read_csv(processed_path)
    current = {feat: df[feat].tail(500).tolist() for feat in CONTINUOUS_FEATURES if feat in df.columns}

    result = detect_drift(baseline_path, current)
    logger.info("Drift detection result: %s", result)
