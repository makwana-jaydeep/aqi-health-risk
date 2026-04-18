import os
import sys
import pytest
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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


def test_health_endpoint():
    response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_ready_endpoint():
    response = requests.get(f"{BACKEND_URL}/api/v1/ready", timeout=10)
    assert response.status_code == 200


def test_predict_returns_valid_response():
    response = requests.post(f"{BACKEND_URL}/api/v1/predict", json=VALID_PAYLOAD, timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "risk_tier" in data
    assert data["risk_tier"] in ["Safe", "Caution", "Avoid Outdoors"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["recommendation"]) > 0


def test_predict_invalid_payload():
    bad_payload = {**VALID_PAYLOAD, "aqi": 999.0}
    response = requests.post(f"{BACKEND_URL}/api/v1/predict", json=bad_payload, timeout=10)
    assert response.status_code == 422


def test_pipeline_status_endpoint():
    response = requests.get(f"{BACKEND_URL}/api/v1/pipeline/status", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data


def test_metrics_endpoint_returns_prometheus_format():
    response = requests.get(f"{BACKEND_URL}/metrics", timeout=10)
    assert response.status_code == 200
    assert "aqi_predictions_total" in response.text
