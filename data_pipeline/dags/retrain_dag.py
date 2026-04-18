import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "aqi-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="model_retraining",
    default_args=DEFAULT_ARGS,
    description="Retrains the AQI risk classification model when drift is detected",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["aqi", "retraining", "mlops"],
) as dag:

    def task_generate_data(**context) -> None:
        import subprocess
        result = subprocess.run(
            ["python", "ml/generate_data.py"],
            cwd="/",
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Data generation failed: {result.stderr}")
        logger.info("Data generation complete: %s", result.stdout)

    def task_run_preprocessing(**context) -> None:
        import subprocess
        result = subprocess.run(
            ["python", "data_pipeline/scripts/preprocess.py"],
            cwd="/",
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Preprocessing failed: {result.stderr}")
        logger.info("Preprocessing complete")

    def task_train_model(**context) -> None:
        import subprocess
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        result = subprocess.run(
            ["python", "ml/train.py"],
            cwd="/",
            env={**os.environ, "MLFLOW_TRACKING_URI": mlflow_uri},
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
        logger.info("Training complete: %s", result.stdout)

    def task_evaluate_model(**context) -> None:
        import subprocess
        result = subprocess.run(
            ["python", "ml/evaluate.py"],
            cwd="/",
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed: {result.stderr}")
        logger.info("Evaluation complete: %s", result.stdout)

    def task_register_model(**context) -> None:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions("aqi_risk_classifier", stages=["None"])
        if versions:
            latest = versions[0]
            client.transition_model_version_stage(
                name="aqi_risk_classifier",
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Model version %s promoted to Production", latest.version)

    generate = PythonOperator(task_id="generate_data", python_callable=task_generate_data)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=task_run_preprocessing)
    train = PythonOperator(task_id="train_model", python_callable=task_train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=task_evaluate_model)
    register = PythonOperator(task_id="register_model", python_callable=task_register_model)

    generate >> preprocess >> train >> evaluate >> register
