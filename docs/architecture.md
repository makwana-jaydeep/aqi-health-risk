# Architecture Document

## AQI Personal Health Risk Assessment

## 1. System Overview

The system is a web-based AI application that classifies personal air quality health risk
(Safe / Caution / Avoid Outdoors) by combining live CPCB pollution data with the user's
health profile. The architecture follows a decoupled, containerized microservice pattern.

## 2. Architecture Diagram (Text Representation)

```
+---------------------+          REST (HTTP)         +---------------------+
|   Streamlit         | <-------------------------> |   FastAPI Backend    |
|   Frontend          |  POST /api/v1/predict        |   (port 8000)        |
|   (port 8501)       |  GET  /api/v1/health         |                      |
+---------------------+  GET  /api/v1/pipeline/status+---------------------+
                                                              |
                          +----------------------------+      |
                          |     ML Model (pkl)         |      |
                          |     GradientBoosting       | <----+
                          |     Classifier             |
                          +----------------------------+
                                    ^
                                    | trains / versions
                          +----------------------------+
                          |   Apache Airflow           |
                          |   (port 8080)              |
                          |   - cpcb_ingestion_dag     |
                          |   - model_retraining_dag   |
                          +----------------------------+
                                    |
              +---------------------+--------------------+
              |                                          |
  +-----------+-----------+              +---------------+----------+
  |   CPCB API            |              |   MLflow Tracking        |
  |   (data.gov.in)       |              |   (port 5000)            |
  |   Hourly AQI Feeds    |              |   Experiments, Models    |
  +-----------------------+              +--------------------------+
                                                  |
              +-----------------------------------+
              |
  +-----------+-----------+              +--------------------------+
  |   Prometheus          |              |   Grafana                |
  |   (port 9090)         |----scrapes-->|   (port 3000)            |
  |   Metrics Store       |              |   Dashboards + Alerts    |
  +-----------------------+              +--------------------------+

  +-----------------------------------------------------------+
  |   DVC                                                     |
  |   Data and Model Version Control                         |
  |   Stages: generate -> preprocess -> train -> evaluate    |
  +-----------------------------------------------------------+

  +-----------------------------------------------------------+
  |   PostgreSQL (internal)                                   |
  |   Airflow metadata store                                  |
  +-----------------------------------------------------------+
```

## 3. Component Descriptions

### 3.1 Streamlit Frontend (port 8501)
A Python web application that provides:
- Risk Assessment page: form-based input for city, pollution readings, and health profile.
- Pipeline Monitor page: live status of ingestion, drift detection, model version, and links
  to Airflow, MLflow, and Grafana.

The frontend communicates with the backend exclusively via REST API calls. It has no direct
access to the model, database, or any data store.

### 3.2 FastAPI Backend (port 8000)
A REST API service that:
- Loads the trained model on startup from a local pickle file or MLflow model registry.
- Exposes POST /api/v1/predict for inference.
- Exposes GET /api/v1/health and GET /api/v1/ready for health checks.
- Exposes GET /api/v1/pipeline/status for pipeline and drift information.
- Exposes GET /metrics in Prometheus exposition format.

The backend enforces strict input validation via Pydantic and implements comprehensive
exception handling with structured logging.

The backend also exposes POST /api/v1/feedback for logging ground truth labels
as they become available. Records are written to feedback_log.jsonl for
periodic performance decay analysis.

### 3.3 ML Model
A GradientBoostingClassifier trained via scikit-learn. The pipeline includes a
StandardScaler followed by the classifier. The model is serialized with joblib and
also registered in the MLflow model registry.

### 3.4 Apache Airflow (port 8080)
Two DAGs manage the automated ML lifecycle:
- **cpcb_ingestion_dag**: Runs hourly. Fetches data from CPCB API (with synthetic
  fallback), validates, preprocesses, computes drift, and conditionally triggers retraining.
- **model_retraining_dag**: Triggered when drift is detected. Regenerates training data,
  preprocesses, trains with MLflow tracking, evaluates, and promotes to Production stage.

### 3.5 MLflow (port 5000)
Tracks every training experiment with parameters, metrics, and artifacts. The trained
model is registered in the MLflow model registry under the name `aqi_risk_classifier`.
The backend can load the Production-stage model directly from the registry.

### 3.6 Prometheus (port 9090)
Scrapes the /metrics endpoint of the backend every 15 seconds. Tracked metrics:
- `aqi_predictions_total`: counter of all predictions.
- `aqi_prediction_duration_seconds`: histogram of inference latency.
- `aqi_risk_tier_total{tier}`: counter per risk tier.
- `aqi_drift_score`: gauge of the latest KS-test drift statistic.

### 3.7 Grafana (port 3000)
Visualizes Prometheus metrics in near-real-time. Provisioned dashboards show:
- Prediction rate and latency percentiles (P50, P95, P99).
- Risk tier distribution over time.
- Drift score with threshold-based coloring.

Alerting rules trigger when error rate exceeds 5% or drift score crosses 0.05.

### 3.8 DVC
Tracks the complete ML pipeline as a DAG of stages:
1. generate_data
2. preprocess
3. train
4. evaluate

Running `dvc repro` from a clean state fully reproduces the pipeline end to end.

## 4. Data Flow

1. Airflow fetches hourly AQI readings from CPCB API.
2. Data is validated, cleaned, and feature-engineered.
3. Drift is assessed using KS-test against baseline statistics.
4. If drift > threshold, the retraining DAG is triggered.
5. A new model is trained, evaluated, and promoted to Production in MLflow.
6. The backend loads the new model on the next health check cycle.
7. User submits a risk assessment via the frontend.
8. Frontend sends a POST request to the backend.
9. Backend runs inference and returns the risk tier.
10. Prometheus records the prediction. Grafana visualizes it.
11. User optionally submits actual outcome via POST /api/v1/feedback.
12. Feedback records accumulate in feedback_log.jsonl for drift and decay analysis.
13. If model performance degrades, scripts/rollback.sh transitions a previous
    MLflow model version back to Production.

## 5. Design Decisions

- **Decoupled frontend and backend**: The frontend and backend run as independent Docker
  containers connected only via REST API. This allows independent scaling and deployment.
- **No cloud**: All infrastructure runs locally or on-premises using Docker Compose.
- **Synthetic data fallback**: When the CPCB API key is absent, the system generates
  statistically realistic synthetic AQI data so the pipeline always has data to process.
- **GradientBoosting over neural networks**: Gradient boosting trains fast on modest hardware,
  is highly interpretable via feature importances, and achieves strong performance on
  tabular data with mixed feature types.
