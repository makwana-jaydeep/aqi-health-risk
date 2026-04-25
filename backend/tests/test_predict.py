import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app
from models.schemas import RiskTier

client = TestClient(app)

VALID_PAYLOAD = {
    "city": "Delhi",
    "aqi": 185.0,
    "pm25": 120.5,
    "pm10": 210.0,
    "no2": 45.0,
    "temperature": 32.0,
    "humidity": 65.0,
    "wind_speed": 8.0,
    "age_group": "adult",
    "has_asthma": False,
    "has_heart_condition": False,
    "planned_activity": "moderate",
}

# methods to check the prediciton of the model 
def test_predict_with_valid_payload_returns_200():
    with patch("api.routes.predict.ModelService.predict") as mock_predict:
        from models.schemas import PredictionResponse
        mock_predict.return_value = PredictionResponse(
            city="Delhi",
            risk_tier=RiskTier.caution,
            risk_score=1,
            confidence=0.82,
            recommendation="",
            model_version="1",
        )
        response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "risk_tier" in data
    assert "confidence" in data
    assert "recommendation" in data


def test_predict_with_invalid_aqi_returns_422():
    payload = {**VALID_PAYLOAD, "aqi": 600.0}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_with_missing_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "city"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_recommendation_is_populated():
    with patch("api.routes.predict.ModelService.predict") as mock_predict:
        from models.schemas import PredictionResponse
        mock_predict.return_value = PredictionResponse(
            city="Mumbai",
            risk_tier=RiskTier.avoid,
            risk_score=2,
            confidence=0.91,
            recommendation="",
            model_version="1",
        )
        payload = {**VALID_PAYLOAD, "city": "Mumbai"}
        response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["recommendation"]) > 0


def test_pipeline_status_returns_200():
    response = client.get("/api/v1/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "model_version" in data
    
def test_feedback_endpoint_returns_200():
    payload = {"city": "Delhi", "actual_risk": 2, "predicted_risk": 1}
    response = client.post("/api/v1/feedback", params=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "logged"
