import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
from scipy import stats

from core.config import settings

logger = logging.getLogger(__name__)

CONTINUOUS_FEATURES = ["aqi", "pm25", "pm10", "no2", "temperature", "humidity", "wind_speed"]


class DriftService:
    _baseline: Dict[str, Any] = {}
    _latest_status: Dict[str, Any] = {
        "drift_detected": False,
        "max_drift_score": 0.0,
        "last_check": "never",
        "feature_scores": {},
    }

    @classmethod
    def load_baseline(cls) -> None:
        path = settings.baseline_path
        if not os.path.exists(path):
            logger.warning("Baseline stats not found at %s", path)
            return
        try:
            with open(path) as f:
                cls._baseline = json.load(f)
            logger.info("Baseline stats loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load baseline stats: %s", exc)

    @classmethod
    def detect_drift(cls, current_samples: Dict[str, list]) -> Dict[str, Any]:
        if not cls._baseline:
            cls.load_baseline()

        if not cls._baseline:
            return cls._latest_status

        feature_scores = {}
        max_score = 0.0
        drift_detected = False

        for feature in CONTINUOUS_FEATURES:
            if feature not in cls._baseline or feature not in current_samples:
                continue
            baseline_data = np.array(cls._baseline[feature]["samples"])
            current_data = np.array(current_samples[feature])

            if len(current_data) < 10:
                continue

            ks_stat, p_value = stats.ks_2samp(baseline_data, current_data)
            feature_scores[feature] = {"ks_stat": float(ks_stat), "p_value": float(p_value)}

            if ks_stat > max_score:
                max_score = ks_stat

            if p_value < 0.05:
                drift_detected = True
                logger.warning("Drift detected in feature=%s ks_stat=%.4f p=%.4f", feature, ks_stat, p_value)

        cls._latest_status = {
            "drift_detected": drift_detected,
            "max_drift_score": max_score,
            "last_check": datetime.utcnow().isoformat(),
            "feature_scores": feature_scores,
        }

        return cls._latest_status

    @classmethod
    def get_latest_drift_status(cls) -> Dict[str, Any]:
        if not cls._baseline:
            cls.load_baseline()
        return cls._latest_status

    @classmethod
    def is_baseline_loaded(cls) -> bool:
        if not cls._baseline:
            cls.load_baseline()
        return bool(cls._baseline)
