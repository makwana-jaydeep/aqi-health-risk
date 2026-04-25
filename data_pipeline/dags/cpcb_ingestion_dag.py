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
# DAG strucuture to get the data and process it
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
        import sys
        sys.path.insert(0, "/opt/airflow/scripts")
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
        import sys
        import os
        sys.path.insert(0, "/opt/airflow/scripts")
        os.chdir("/opt/airflow")

        import pandas as pd
        import numpy as np

        raw_path = "/data/raw"
        processed_dir = "/data/processed"
        baseline_path = "/data/baseline_stats.json"
        os.makedirs(processed_dir, exist_ok=True)

        import glob
        csv_files = sorted(glob.glob(f"{raw_path}/*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path}")

        df = pd.read_csv(csv_files[-1])

        from feature_engineering import engineer_features
        df = df.dropna(subset=["aqi", "pm25", "pm10"])
        df = df[df["aqi"].between(0, 500)]
        df = engineer_features(df)

        df.to_csv(f"{processed_dir}/aqi_processed.csv", index=False)

        import json
        from scipy import stats
        baseline = {}
        for feat in ["aqi", "pm25", "pm10", "no2", "temperature", "humidity", "wind_speed"]:
            if feat in df.columns:
                vals = df[feat].dropna().tolist()
                baseline[feat] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "samples": vals[:2000],
                }
        with open(baseline_path, "w") as f:
            json.dump(baseline, f)

    def task_check_drift(**context) -> str:
        import json
        import os
        import sys
        import numpy as np
        import pandas as pd
        from scipy import stats

        baseline_path = "/data/baseline_stats.json"
        processed_path = "/data/processed/aqi_processed.csv"

        if not os.path.exists(baseline_path):
            return "no_drift"
        if not os.path.exists(processed_path):
            return "no_drift"

        with open(baseline_path) as f:
            baseline = json.load(f)

        df = pd.read_csv(processed_path)
        features = ["aqi", "pm25", "pm10", "no2", "temperature", "humidity", "wind_speed"]

        for feat in features:
            if feat not in baseline or feat not in df.columns:
                continue
            baseline_samples = np.array(baseline[feat]["samples"])
            current_samples = df[feat].dropna().tail(500).values
            if len(current_samples) < 10:
                continue
            _, p_value = stats.ks_2samp(baseline_samples, current_samples)
            if p_value < 0.05:
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
