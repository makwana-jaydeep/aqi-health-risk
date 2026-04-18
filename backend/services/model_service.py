import json
import logging
import os
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from core.config import settings
from models.schemas import PredictionRequest, PredictionResponse, RiskTier

logger = logging.getLogger(__name__)

AGE_GROUP_MAP = {"child": 0, "teen": 1, "adult": 2, "senior": 3}
ACTIVITY_MAP = {"low": 0, "moderate": 1, "high": 2}
RISK_LABELS = {0: RiskTier.safe, 1: RiskTier.caution, 2: RiskTier.avoid}

FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "no2",
    "temperature", "humidity", "wind_speed",
    "age_group_enc", "has_asthma", "has_heart_condition", "planned_activity_enc",
]


class ModelService:
    _model: Optional[Pipeline] = None
    _version: str = "unknown"

    @classmethod
    def load_model(cls) -> None:
        model_path = settings.model_path
        if os.path.exists(model_path):
            try:
                cls._model = joblib.load(model_path)
                cls._version = _read_version(model_path)
                logger.info("Model loaded from %s version=%s", model_path, cls._version)
                return
            except Exception as exc:
                logger.warning("Could not load model from %s: %s", model_path, exc)

        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(settings.model_name, stages=["Production"])
            if versions:
                mv = versions[0]
                cls._model = mlflow.sklearn.load_model(mv.source)
                cls._version = mv.version
                logger.info("Model loaded from MLflow registry version=%s", cls._version)
                return
        except Exception as exc:
            logger.warning("Could not load model from MLflow: %s", exc)

        logger.error("No model available. Run dvc repro to train a model first.")

    @classmethod
    def predict(cls, request: PredictionRequest) -> PredictionResponse:
        if cls._model is None:
            raise RuntimeError("Model is not loaded")

        features = _build_features(request)
        df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        proba = cls._model.predict_proba(df)[0]
        risk_score = int(np.argmax(proba))
        confidence = float(proba[risk_score])
        risk_tier = RISK_LABELS[risk_score]

        return PredictionResponse(
            city=request.city,
            risk_tier=risk_tier,
            risk_score=risk_score,
            confidence=confidence,
            recommendation="",
            model_version=cls._version,
        )

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def get_version(cls) -> str:
        return cls._version


def _build_features(request: PredictionRequest) -> list:
    return [
        request.aqi,
        request.pm25,
        request.pm10,
        request.no2,
        request.temperature,
        request.humidity,
        request.wind_speed,
        AGE_GROUP_MAP[request.age_group.value],
        int(request.has_asthma),
        int(request.has_heart_condition),
        ACTIVITY_MAP[request.planned_activity.value],
    ]


def _read_version(model_path: str) -> str:
    version_file = model_path.replace(".pkl", "_version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return "local-v1"
