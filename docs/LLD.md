# Low-Level Design Document

## AQI Personal Health Risk Assessment

## 1. API Endpoint Specifications

### Base URL
```
http://localhost:8000
```

---

### POST /api/v1/predict

**Description**: Accepts city, pollution readings, weather, and user health profile.
Returns a personalized risk tier classification.

**Request Body** (application/json):
```json
{
  "city": "string",
  "aqi": "number [0, 500]",
  "pm25": "number [>= 0]",
  "pm10": "number [>= 0]",
  "no2": "number [>= 0]",
  "temperature": "number",
  "humidity": "number [0, 100]",
  "wind_speed": "number [>= 0]",
  "age_group": "string (child | teen | adult | senior)",
  "has_asthma": "boolean",
  "has_heart_condition": "boolean",
  "planned_activity": "string (low | moderate | high)"
}
```

**Response 200** (application/json):
```json
{
  "city": "string",
  "risk_tier": "string (Safe | Caution | Avoid Outdoors)",
  "risk_score": "integer (0 | 1 | 2)",
  "confidence": "number [0.0, 1.0]",
  "recommendation": "string",
  "model_version": "string"
}
```

**Response 422** (Validation Error):
```json
{
  "detail": [{"loc": [...], "msg": "...", "type": "..."}]
}
```

**Response 503** (Model Not Available):
```json
{"detail": "Model not available"}
```

---

### GET /api/v1/health

**Description**: Liveness check. Returns API status and whether the model is loaded.

**Response 200**:
```json
{
  "status": "ok",
  "model_loaded": "boolean",
  "version": "string"
}
```

---

### GET /api/v1/ready

**Description**: Readiness check for orchestration (k8s / Docker health checks).

**Response 200**:
```json
{
  "ready": "boolean",
  "model_loaded": "boolean",
  "baseline_loaded": "boolean"
}
```

---

### GET /api/v1/pipeline/status

**Description**: Returns current pipeline state, drift status, and prediction counters.

**Response 200**:
```json
{
  "last_ingestion": "string (ISO 8601 timestamp)",
  "drift_detected": "boolean",
  "drift_score": "number",
  "model_version": "string",
  "total_predictions": "integer"
}
```

---

### GET /metrics

**Description**: Prometheus-format metrics exposition.

**Response 200** (text/plain):
```
# HELP aqi_predictions_total Total number of AQI risk predictions served
# TYPE aqi_predictions_total counter
aqi_predictions_total 142.0
...
```

---

## 2. Module Structure

### backend/
```
main.py                         FastAPI application factory and startup
core/
  config.py                     Pydantic settings from env vars
  logging_config.py             Structured logging setup
models/
  schemas.py                    Pydantic request/response models, enums
api/routes/
  predict.py                    POST /predict handler
  health.py                     GET /health, GET /ready handlers
  pipeline.py                   GET /pipeline/status handler
services/
  model_service.py              Model loading, feature preprocessing, inference
  drift_service.py              KS-test drift detection against baseline
  prometheus_service.py         Prometheus Counter, Histogram, Gauge definitions
tests/
  test_predict.py               Unit tests for prediction endpoint
  test_health.py                Unit tests for health endpoints
  test_model_service.py         Unit tests for ModelService
```

### ml/
```
generate_data.py                Synthetic AQI dataset generator
train.py                        Training script with MLflow integration
evaluate.py                     Evaluation script producing classification report
drift_detection.py              KS-test drift utilities and baseline computation
```

### data_pipeline/
```
scripts/
  ingest_cpcb.py                CPCB API fetch with synthetic fallback
  preprocess.py                 Cleaning, outlier removal, baseline generation
  feature_engineering.py        Derived feature computation
dags/
  cpcb_ingestion_dag.py         Airflow DAG: fetch -> validate -> preprocess -> drift check
  retrain_dag.py                Airflow DAG: generate -> preprocess -> train -> register
```

### frontend/
```
app.py                          Streamlit multipage app
```

---

## 3. Data Schemas

### Internal Feature Vector (model input)
```
[aqi, pm25, pm10, no2, temperature, humidity, wind_speed,
 age_group_enc, has_asthma, has_heart_condition, planned_activity_enc]
```

Shape: (1, 11)

### Baseline Statistics File (JSON)
```json
{
  "aqi": {
    "mean": 142.3,
    "std": 62.1,
    "min": 8.0,
    "max": 498.0,
    "p25": 94.2,
    "p75": 194.6,
    "samples": [...]
  },
  ...
}
```

---

## 4. Service Environment Variables

| Variable              | Default                  | Description                    |
|-----------------------|--------------------------|--------------------------------|
| MLFLOW_TRACKING_URI   | http://localhost:5000    | MLflow server URL              |
| MODEL_PATH            | data/model.pkl           | Path to local model file       |
| BASELINE_PATH         | data/baseline_stats.json | Path to baseline stats file    |
| LOG_LEVEL             | INFO                     | Python logging level           |
| BACKEND_URL           | http://localhost:8000    | Backend URL (frontend use)     |
| CPCB_API_KEY          | (empty)                  | CPCB data.gov.in API key       |

---

## 5. Logging Format

All log entries follow this structured format:
```
2024-01-15 14:30:22 | INFO     | services.model_service | Model loaded from data/model.pkl version=3a9f1b2c
```

Fields: timestamp, level, logger name, message.

---

## 6. Exception Handling Strategy

| Exception Type          | HTTP Status | Behaviour                              |
|-------------------------|-------------|----------------------------------------|
| Pydantic ValidationError| 422         | Automatic FastAPI validation           |
| RuntimeError (no model) | 503         | Logged as ERROR, returns 503           |
| Any unexpected Exception| 500         | Logged via logger.exception, returns 500 |
| Requests Timeout        | N/A         | Airflow task retries (max 2)           |
| CPCB API failure        | N/A         | Falls back to synthetic data generator |
