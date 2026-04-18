# High-Level Design Document

## AQI Personal Health Risk Assessment

## 1. Problem Statement

Urban Indians face air quality that is frequently hazardous, yet government tools show only
a raw AQI number with a static color band. A 60-year-old with asthma faces a different risk
from the same reading as a healthy 25-year-old. This application personalizes AQI risk to
the individual and provides actionable health guidance.

## 2. Goals and Non-Goals

### Goals
- Classify personal health risk as Safe, Caution, or Avoid Outdoors.
- Serve recommendations in under 200ms P95 latency.
- Automate the full data ingestion and retraining pipeline.
- Detect data drift (monsoon, Diwali, crop-burning seasons) and retrain automatically.
- Track every experiment with full reproducibility.
- Provide a dashboard for non-technical users and a monitoring console for operators.

### Non-Goals
- Provide medical advice (the tool is informational only).
- Deploy on cloud infrastructure (all services run on-premises).
- Build a native mobile app.

## 3. ML System Design

### 3.1 Model Choice
A **GradientBoostingClassifier** (sklearn) is selected because:
- It handles mixed feature types (continuous + categorical) natively.
- It achieves high accuracy on tabular data without hyperparameter sensitivity.
- It produces calibrated probabilities for the confidence score.
- It trains within minutes on commodity hardware.
- Feature importances are directly available for explainability.

### 3.2 Input Features
| Feature              | Type        | Description                              |
|----------------------|-------------|------------------------------------------|
| aqi                  | float       | Overall AQI (0-500)                      |
| pm25                 | float       | PM2.5 concentration (ug/m3)              |
| pm10                 | float       | PM10 concentration (ug/m3)               |
| no2                  | float       | NO2 concentration (ug/m3)                |
| temperature          | float       | Temperature in Celsius                   |
| humidity             | float       | Relative humidity (%)                    |
| wind_speed           | float       | Wind speed (km/h)                        |
| age_group_enc        | int (0-3)   | child=0, teen=1, adult=2, senior=3       |
| has_asthma           | int (0/1)   | Asthma or respiratory condition          |
| has_heart_condition  | int (0/1)   | Cardiovascular condition                 |
| planned_activity_enc | int (0-2)   | low=0, moderate=1, high=2                |

### 3.3 Target Classes
| Class | Label          | Description                              |
|-------|----------------|------------------------------------------|
| 0     | Safe           | Normal outdoor activity acceptable       |
| 1     | Caution        | Sensitive individuals should limit exposure |
| 2     | Avoid Outdoors | Hazardous; stay indoors                  |

### 3.4 Evaluation Metrics
- Primary: **F1-Macro** (balances across imbalanced classes)
- Secondary: Accuracy, F1-Weighted
- Business: Inference latency P95 < 200ms

## 4. MLOps Pipeline Design

### 4.1 Data Engineering
The CPCB API (data.gov.in) publishes hourly AQI data for 300+ Indian cities. Airflow
ingests this data, validates schema and null rates, removes outliers, and engineers
derived features. A synthetic data generator provides fallback when the API is unavailable.

### 4.2 Experiment Tracking
Every training run is logged to MLflow with:
- Parameters: n_estimators, max_depth, learning_rate, random_state, train_samples
- Metrics: accuracy, f1_macro, f1_weighted
- Artifacts: serialized pipeline, feature importance plot

### 4.3 Drift Detection
After each ingestion cycle, a Kolmogorov-Smirnov test compares the current window of
incoming data against the baseline computed at initial training time. A p-value below 0.05
on any continuous feature flags drift and triggers the retraining DAG.

### 4.4 Continuous Integration
DVC defines the pipeline as a DAG:
```
generate_data --> preprocess --> train --> evaluate
```
`dvc repro` detects which stages are stale and reruns only those stages.

## 5. Service Architecture Summary

| Service    | Technology    | Responsibility                             |
|------------|---------------|--------------------------------------------|
| Frontend   | Streamlit     | User interface, form submission, results   |
| Backend    | FastAPI       | REST API, inference, metrics exposure      |
| Pipeline   | Airflow       | Data ingestion, drift check, retraining    |
| Tracking   | MLflow        | Experiment logging, model registry         |
| Monitoring | Prometheus    | Time-series metrics collection             |
| Dashboard  | Grafana       | Visualization and alerting                 |
| Versioning | DVC + Git     | Code, data, and model version control      |
| Containers | Docker Compose| Environment parity across all services    |

## 6. Security Considerations

- All sensitive data (CPCB API keys) are passed via environment variables, never hardcoded.
- The backend does not log raw user input.
- Health check endpoints do not expose internal implementation details.
- Grafana admin password is set via Docker environment variable (change in production).
