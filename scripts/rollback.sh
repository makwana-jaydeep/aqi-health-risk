#!/bin/bash
# Rollback to the previous MLflow Production model version
VERSION=${1:-""}
if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/rollback.sh <version_number>"
    exit 1
fi
python3 - << EOF
import mlflow
mlflow.set_tracking_uri("file:///$(pwd)/mlruns")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="aqi_risk_classifier",
    version="$VERSION",
    stage="Production",
    archive_existing_versions=True,
)
print(f"Rolled back to version $VERSION")
EOF