import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from models.schemas import (
    ActivityLevel,
    AgeGroup,
    PredictionRequest,
    RiskTier,
)
from services.model_service import ModelService, _build_features

# methods to check model api working 
def make_request(**kwargs) -> PredictionRequest:
    defaults = dict(
        city="Delhi",
        aqi=150.0,
        pm25=90.0,
        pm10=180.0,
        no2=40.0,
        temperature=30.0,
        humidity=60.0,
        wind_speed=10.0,
        age_group=AgeGroup.adult,
        has_asthma=False,
        has_heart_condition=False,
        planned_activity=ActivityLevel.moderate,
    )
    defaults.update(kwargs)
    return PredictionRequest(**defaults)


def test_build_features_returns_correct_length():
    req = make_request()
    features = _build_features(req)
    assert len(features) == 11


def test_build_features_age_group_encoding():
    req_child = make_request(age_group=AgeGroup.child)
    req_senior = make_request(age_group=AgeGroup.senior)
    assert _build_features(req_child)[7] == 0
    assert _build_features(req_senior)[7] == 3


def test_build_features_activity_encoding():
    req_low = make_request(planned_activity=ActivityLevel.low)
    req_high = make_request(planned_activity=ActivityLevel.high)
    assert _build_features(req_low)[10] == 0
    assert _build_features(req_high)[10] == 2


def test_build_features_boolean_conditions():
    req = make_request(has_asthma=True, has_heart_condition=True)
    features = _build_features(req)
    assert features[8] == 1
    assert features[9] == 1


def test_predict_raises_when_model_not_loaded():
    original = ModelService._model
    ModelService._model = None
    with pytest.raises(RuntimeError, match="not loaded"):
        ModelService.predict(make_request())
    ModelService._model = original


def test_is_loaded_returns_false_when_no_model():
    original = ModelService._model
    ModelService._model = None
    assert ModelService.is_loaded() is False
    ModelService._model = original
