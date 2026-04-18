import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_path: str = os.getenv("MODEL_PATH", "data/model.pkl")
    baseline_path: str = os.getenv("BASELINE_PATH", "data/baseline_stats.json")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_name: str = "aqi_risk_classifier"
    app_version: str = "1.0.0"

    model_config = ConfigDict(env_file=".env")


settings = Settings()
