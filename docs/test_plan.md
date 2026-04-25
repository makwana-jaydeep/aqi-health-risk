# Test Plan

## AQI Personal Health Risk Assessment

## 1. Overview

This document defines the testing strategy, test cases, and acceptance criteria for
the AQI Personal Health Risk Assessment application.

## 2. Testing Levels

### 2.1 Unit Tests
Tests for individual functions and classes in isolation, using mocks where dependencies
are external. Located in `backend/tests/` and `tests/unit/`.

Tools: pytest, unittest.mock

### 2.2 Integration Tests
Tests that verify the full API request-response cycle with the backend running.
Located in `tests/integration/`.

Tools: pytest, requests

### 2.3 End-to-End Tests (Manual)
Performed manually by accessing the Streamlit UI at http://localhost:8501 and verifying
the full user journey from input to recommendation display.

## 3. Test Cases

### Unit: Feature Engineering
| ID    | Description                                 | Expected Result      |
|-------|---------------------------------------------|----------------------|
| TC-U01| engineer_features adds pm25_aqi_ratio       | Column present       |
| TC-U02| engineer_features adds heat_index           | Column present       |
| TC-U03| low_wind=1 when wind_speed < 5              | low_wind equals 1    |
| TC-U04| aqi_severity computed from AQI bins         | Values in [0,4]      |
| TC-U05| Original DataFrame not modified             | No new columns in df |
| TC-U06| Zero AQI does not produce NaN               | No NaN in output     |

### Unit: Drift Detection
| ID    | Description                                 | Expected Result      |
|-------|---------------------------------------------|----------------------|
| TC-U07| compute_baseline_stats writes JSON file     | File exists, valid   |
| TC-U08| No drift for same distribution              | drift_detected=False |
| TC-U09| Drift detected for large distribution shift | drift_detected=True  |

### Unit: Model Service
| ID    | Description                                 | Expected Result      |
|-------|---------------------------------------------|----------------------|
| TC-U10| _build_features returns 11 features         | len==11              |
| TC-U11| child age_group encoded as 0                | features[7]==0       |
| TC-U12| senior age_group encoded as 3               | features[7]==3       |
| TC-U13| high activity encoded as 2                  | features[10]==2      |
| TC-U14| has_asthma=True encoded as 1                | features[8]==1       |
| TC-U15| predict raises RuntimeError with no model   | RuntimeError raised  |
| TC-U16| is_loaded returns False with no model       | False                |

### Unit: API Endpoints
| ID    | Description                                 | Expected Result      |
|-------|---------------------------------------------|----------------------|
| TC-U17| /health returns 200 with status ok          | 200 + status=="ok"   |
| TC-U18| /ready returns 200 with bool flags          | 200 + bool fields    |
| TC-U19| /metrics returns 200                        | 200 + text/plain     |
| TC-U20| /predict valid payload returns 200          | 200 + risk_tier      |
| TC-U21| /predict aqi>500 returns 422                | 422 validation error |
| TC-U22| /predict missing city returns 422           | 422 validation error |
| TC-U23| /predict recommendation non-empty           | len > 0              |
| TC-U24| /pipeline/status returns 200                | 200 + drift fields   |
| TC-U25| /feedback logs ground truth record          | 200 + status logged  |
| TC-U26| /feedback with invalid risk value returns 422 | 422                |

### Integration
| ID    | Description                                 | Expected Result      |
|-------|---------------------------------------------|----------------------|
| TC-I01| /health over HTTP returns ok                | 200 + ok status      |
| TC-I02| /predict end-to-end returns valid tier      | Valid tier + conf    |
| TC-I03| /predict invalid AQI returns 422            | 422                  |
| TC-I04| /pipeline/status returns drift fields       | Valid JSON           |
| TC-I05| /metrics contains expected metric names     | Metric names present |

## 4. Acceptance Criteria

| ID  | Criterion                                              | Threshold        |
|-----|--------------------------------------------------------|------------------|
| AC1 | Model accuracy on held-out test set                    | >= 85%           |
| AC2 | F1-Macro on held-out test set                          | >= 0.80          |
| AC3 | P95 prediction latency                                 | < 200ms          |
| AC4 | API availability (health endpoint)                     | 200 OK           |
| AC5 | Drift detection fires for large distribution shift     | drift_detected=True |
| AC6 | DVC repro runs end-to-end from clean state             | Exit code 0      |
| AC7 | All unit tests pass                                    | 0 failures       |
| AC8 | All Docker services start and pass health checks       | All healthy      |
| AC9 | Rollback script transitions model version in MLflow | Stage = Production |
| AC10| Feedback endpoint appends to feedback_log.jsonl    | File grows on POST |

## 5. Running the Test Suite

```bash
cd backend
python -m pytest tests/ -v --tb=short

cd ..
python -m pytest tests/ -v --tb=short

python -m pytest tests/unit/ tests/integration/ -v
```

## 6. Known Limitations

- Integration tests require a running backend. Run with `docker compose up backend` first.
- CPCB live data requires a valid API key set in the CPCB_API_KEY environment variable.
  Without this, all ingestion uses the synthetic data fallback, which is still suitable
  for all MLOps pipeline tests.
