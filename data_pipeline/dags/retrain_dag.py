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
# DAG to retrain the model on newly added data
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
        import sys
        import numpy as np
        import pandas as pd
        import os
        sys.path.insert(0, "/opt/airflow/scripts")

        rng = np.random.default_rng(42)
        cities = ["Delhi","Mumbai","Chennai","Bengaluru","Kolkata","Hyderabad","Pune","Ahmedabad"]
        profiles = {"Delhi":(200,80),"Mumbai":(130,50),"Chennai":(100,40),
                    "Bengaluru":(90,35),"Kolkata":(160,60),"Hyderabad":(110,45),
                    "Pune":(105,40),"Ahmedabad":(145,55)}
        records = []
        for _ in range(15000):
            city = cities[rng.integers(0, len(cities))]
            m, s = profiles[city]
            aqi = float(np.clip(rng.normal(m, s), 5, 500))
            records.append({
                "city": city, "aqi": round(aqi,2),
                "pm25": round(float(np.clip(aqi*rng.uniform(0.55,0.75),0,400)),2),
                "pm10": round(float(np.clip(aqi*rng.uniform(0.9,1.3),0,500)),2),
                "no2": round(float(np.clip(rng.normal(30,15),0,200)),2),
                "temperature": round(float(rng.uniform(15,45)),2),
                "humidity": round(float(rng.uniform(20,95)),2),
                "wind_speed": round(float(np.clip(rng.exponential(10),0,60)),2),
                "age_group_enc": int(rng.integers(0,4)),
                "has_asthma": int(rng.random()<0.12),
                "has_heart_condition": int(rng.random()<0.08),
                "planned_activity_enc": int(rng.integers(0,3)),
                "risk_tier": int(min(2 if aqi>150 else 1 if aqi>50 else 0, 2)),
            })
        os.makedirs("/data/raw", exist_ok=True)
        pd.DataFrame(records).to_csv("/data/raw/aqi_dataset.csv", index=False)
        logger.info("Generated 15000 samples to /data/raw/aqi_dataset.csv")


    def task_run_preprocessing(**context) -> None:
        import sys
        import os
        import json
        import numpy as np
        import pandas as pd
        sys.path.insert(0, "/opt/airflow/scripts")
        from feature_engineering import engineer_features

        df = pd.read_csv("/data/raw/aqi_dataset.csv")
        df = df.dropna(subset=["aqi","pm25","pm10"])
        df = df[df["aqi"].between(0,500)]
        df = engineer_features(df)

        os.makedirs("/data/processed", exist_ok=True)
        df.to_csv("/data/processed/aqi_processed.csv", index=False)

        baseline = {}
        for feat in ["aqi","pm25","pm10","no2","temperature","humidity","wind_speed"]:
            if feat in df.columns:
                vals = df[feat].dropna().tolist()
                baseline[feat] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "samples": vals[:2000]}
        with open("/data/baseline_stats.json","w") as f:
            json.dump(baseline, f)
        logger.info("Preprocessing complete")


    def task_train_model(**context) -> None:
        import sys
        import os
        import json
        import joblib
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        FEATURES = ["aqi","pm25","pm10","no2","temperature","humidity","wind_speed",
                    "age_group_enc","has_asthma","has_heart_condition","planned_activity_enc"]

        df = pd.read_csv("/data/processed/aqi_processed.csv")
        X = df[FEATURES]
        y = df["risk_tier"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("aqi-risk-classification")

        with mlflow.start_run() as run:
            pipeline = Pipeline([("scaler", StandardScaler()),
                                ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42))])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro"))
            mlflow.log_params({"n_estimators":200,"max_depth":6})
            mlflow.log_metrics({"accuracy":acc,"f1_macro":f1})
            mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name="aqi_risk_classifier")

        joblib.dump(pipeline, "/data/model.pkl")
        with open("/data/model_version.txt","w") as f:
            f.write(run.info.run_id[:8])
        logger.info("Training complete accuracy=%.4f f1=%.4f", acc, f1)


    def task_evaluate_model(**context) -> None:
        import joblib
        import json
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        FEATURES = ["aqi","pm25","pm10","no2","temperature","humidity","wind_speed",
                    "age_group_enc","has_asthma","has_heart_condition","planned_activity_enc"]

        model = joblib.load("/data/model.pkl")
        df = pd.read_csv("/data/processed/aqi_processed.csv")
        X = df[FEATURES]
        y = df["risk_tier"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, y_pred)), "f1_macro": float(f1_score(y_test, y_pred, average="macro"))}
        with open("/data/eval_metrics.json","w") as f:
            json.dump(metrics, f)
        logger.info("Evaluation complete: %s", metrics)


    def task_register_model(**context) -> None:
        import os
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("aqi_risk_classifier", stages=["None"])
        if versions:
            client.transition_model_version_stage(
                name="aqi_risk_classifier",
                version=versions[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Model version %s promoted to Production", versions[0].version)

    generate = PythonOperator(task_id="generate_data", python_callable=task_generate_data)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=task_run_preprocessing)
    train = PythonOperator(task_id="train_model", python_callable=task_train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=task_evaluate_model)
    register = PythonOperator(task_id="register_model", python_callable=task_register_model)

    generate >> preprocess >> train >> evaluate >> register
