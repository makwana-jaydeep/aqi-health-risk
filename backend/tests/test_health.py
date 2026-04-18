import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "version" in data


def test_ready_endpoint_returns_status():
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "model_loaded" in data
    assert "baseline_loaded" in data


def test_metrics_endpoint_accessible():
    response = client.get("/metrics")
    assert response.status_code == 200
