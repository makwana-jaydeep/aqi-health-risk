import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, "/opt/airflow/scripts")

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "aqi-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="cpcb_data_ingestion",
    default_args=DEFAULT_ARGS,
    description="Hourly CPCB AQI data ingestion, preprocessing, and drift detection",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["aqi", "ingestion", "mlops"],
) as dag:

    def task_fetch_data(**context) -> str:
        from ingest_cpcb import ingest
        output_path = ingest(output_dir="/data/raw")
        context["ti"].xcom_push(key="raw_path", value=output_path)
        return output_path

    def task_validate_data(**context) -> bool:
        import pandas as pd
        raw_path = context["ti"].xcom_pull(key="raw_path", task_ids="fetch_data")
        df = pd.read_csv(raw_path)

        issues = []
        required_cols = ["aqi", "pm25", "pm10", "no2", "temperature", "humidity", "wind_speed"]
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Missing column: {col}")

        null_pct = df[required_cols].isnull().mean()
        high_null = null_pct[null_pct > 0.3]
        if not high_null.empty:
            issues.append(f"High null rate in: {high_null.index.tolist()}")

        if issues:
            raise ValueError(f"Data validation failed: {issues}")

        logger.info("Data validation passed for %s rows=%d", raw_path, len(df))
        return True

    def task_preprocess(**context) -> None:
        import pandas as pd
        import sys
        sys.path.insert(0, "/opt/airflow/scripts")
        sys.path.insert(0, "/opt/airflow")
        from preprocess import main
        main()

    def task_check_drift(**context) -> str:
        import json
        import os
        import pandas as pd
        import sys
        sys.path.insert(0, "/opt/airflow/scripts")

        baseline_path = "/data/baseline_stats.json"
        processed_path = "/data/processed/aqi_processed.csv"

        if not os.path.exists(baseline_path):
            logger.warning("Baseline not found, skipping drift check")
            return "no_drift"

        from drift_detection import detect_drift, CONTINUOUS_FEATURES
        df = pd.read_csv(processed_path)
        current = {f: df[f].tail(500).tolist() for f in CONTINUOUS_FEATURES if f in df.columns}
        result = detect_drift(baseline_path, current)

        if result["drift_detected"]:
            logger.warning("Drift detected. Triggering retraining.")
            return "drift_detected"
        return "no_drift"

    fetch = PythonOperator(
        task_id="fetch_data",
        python_callable=task_fetch_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess,
    )

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
    )

    trigger_retrain = PythonOperator(
        task_id="drift_detected",
        python_callable=lambda: logger.info("Drift detected - retraining DAG should be triggered"),
    )

    no_drift = EmptyOperator(task_id="no_drift")

    done = EmptyOperator(task_id="done", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    fetch >> validate >> preprocess >> check_drift >> [trigger_retrain, no_drift] >> done
