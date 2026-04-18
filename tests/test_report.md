# Test Report

## Project: AQI Personal Health Risk Assessment

## Test Summary

| Category           | Total | Passed | Failed |
|--------------------|-------|--------|--------|
| Unit Tests         | 23    | 23     | 0      |
| Integration Tests  | 6     | 6      | 0      |
| **Total**          | **29**| **29** | **0**  |

## Acceptance Criteria

| ID  | Criterion                                          | Status |
|-----|----------------------------------------------------|--------|
| AC1 | API responds with valid risk tier for all inputs   | PASS   |
| AC2 | Prediction latency P95 < 200ms                     | PASS   |
| AC3 | Model accuracy >= 85% on test set                  | PASS   |
| AC4 | Drift detection alerts when KS stat > 0.05         | PASS   |
| AC5 | Frontend displays risk clearly for non-tech users  | PASS   |
| AC6 | All Docker services start cleanly                  | PASS   |
| AC7 | DVC pipeline is reproducible from clean state      | PASS   |
| AC8 | MLflow tracks every experiment with params/metrics | PASS   |

## Unit Test Cases

### Feature Engineering (tests/unit/test_feature_engineering.py)
- TC-U01: pm25_aqi_ratio column added by engineer_features
- TC-U02: heat_index column added by engineer_features
- TC-U03: low_wind flag set correctly for wind_speed < 5 km/h
- TC-U04: aqi_severity categorical computed correctly
- TC-U05: engineer_features does not modify original DataFrame
- TC-U06: engineer_features handles zero AQI without NaN

### Drift Detection (tests/unit/test_drift_detection.py)
- TC-U07: compute_baseline_stats creates a valid JSON file
- TC-U08: detect_drift returns no drift for identical distributions
- TC-U09: detect_drift returns drift for significantly shifted distributions

### Backend API (backend/tests/test_health.py)
- TC-U10: /health returns 200 with status ok
- TC-U11: /ready returns 200 with readiness flags
- TC-U12: /metrics returns 200 (Prometheus format)

### Backend Prediction (backend/tests/test_predict.py)
- TC-U13: /predict returns 200 for valid payload
- TC-U14: /predict returns 422 for AQI > 500
- TC-U15: /predict returns 422 for missing required field
- TC-U16: /predict recommendation field is non-empty
- TC-U17: /pipeline/status returns 200

### Model Service (backend/tests/test_model_service.py)
- TC-U18: _build_features returns 11 features
- TC-U19: age_group encoding is correct
- TC-U20: activity level encoding is correct
- TC-U21: boolean conditions encoded as 0/1
- TC-U22: predict raises RuntimeError when model not loaded
- TC-U23: is_loaded returns False when model is None

## Integration Test Cases (tests/integration/test_api_integration.py)

- TC-I01: Health endpoint returns ok
- TC-I02: Ready endpoint returns status
- TC-I03: Predict returns valid risk tier
- TC-I04: Invalid AQI returns 422
- TC-I05: Pipeline status endpoint returns data
- TC-I06: Prometheus metrics endpoint accessible

## Test Environment

- Python 3.11
- pytest 8.2.0
- pytest-asyncio 0.23.6
- All tests run inside Docker containers or virtualenv

## Running Tests

```bash
cd backend && python -m pytest tests/ -v --tb=short
python -m pytest tests/ -v --tb=short
```
