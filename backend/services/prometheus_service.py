from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNTER = Counter(
    "aqi_predictions_total",
    "Total number of AQI risk predictions served",
)

PREDICTION_DURATION = Histogram(
    "aqi_prediction_duration_seconds",
    "Time taken to produce a prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

RISK_TIER_COUNTER = Counter(
    "aqi_risk_tier_total",
    "Total predictions by risk tier",
    ["tier"],
)

DRIFT_SCORE_GAUGE = Gauge(
    "aqi_drift_score",
    "Latest KS-test drift score for input features",
)

MODEL_VERSION_INFO = Gauge(
    "aqi_model_version_info",
    "Current model version loaded",
    ["version"],
)
