# AQI Personal Health Risk Assessment

A production-grade AI application that delivers personalized air quality health risk
recommendations for Indian cities. The system classifies personal risk as Safe, Caution,
or Avoid Outdoors based on live pollution data and the user's health profile.

## Architecture Overview

- **Frontend**: Streamlit web application (port 8501)
- **Backend**: FastAPI REST API (port 8000)
- **Data Pipeline**: Apache Airflow (port 8080)
- **Experiment Tracking**: MLflow (port 5000)
- **Monitoring**: Prometheus (port 9090) + Grafana (port 3000)
- **Database**: PostgreSQL (Airflow metadata)

## Quick Start

### Prerequisites
- Docker >= 24.0 and Docker Compose >= 2.0
- Python 3.11+
- Git
- DVC (`pip install dvc`)

### Option 1: Docker Compose (Recommended)

```bash
git clone <repo-url>
cd aqi-health-risk
chmod +x setup.sh && ./setup.sh
docker compose up --build
```

Open http://localhost:8501 to access the application.

### Option 2: Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
dvc repro
cd backend && uvicorn main:app --reload --port 8000 &
cd ../frontend && streamlit run app.py
```

## Service URLs

| Service    | URL                        | Credentials |
|------------|----------------------------|-------------|
| Frontend   | http://localhost:8501      | -           |
| API Docs   | http://localhost:8000/docs | -           |
| Airflow    | http://localhost:8080      | admin/admin |
| MLflow     | http://localhost:5000      | -           |
| Prometheus | http://localhost:9090      | -           |
| Grafana    | http://localhost:3000      | admin/admin |

## DVC Pipeline

```bash
dvc repro          # Run the full pipeline
dvc dag            # Visualize the pipeline DAG
dvc params diff    # Show parameter changes
dvc metrics show   # Display model metrics
```

## Running Tests

```bash
cd backend && python -m pytest tests/ -v --tb=short
python -m pytest tests/ -v --tb=short
```
## MLproject — Reproducible Training

Run training as an MLflow project (links Git commit to experiment run):

```bash
mlflow run . -e main --env-manager=local
```

## Rollback a Model Version

If a deployed model needs to be rolled back:

```bash
./scripts/rollback.sh <version_number>
docker compose restart backend
```

Check available versions at http://localhost:5000 under Models.

## Documentation

- `docs/architecture.md` - Architecture diagram and component descriptions
- `docs/HLD.md` - High-level design document
- `docs/LLD.md` - Low-level design with API specifications
- `docs/test_plan.md` - Test plan and test cases
- `docs/user_manual.md` - Non-technical user guide
