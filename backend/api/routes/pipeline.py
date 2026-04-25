import logging

from fastapi import APIRouter

from models.schemas import PipelineStatusResponse
from services.drift_service import DriftService
from services.model_service import ModelService
from services.prometheus_service import PREDICTION_COUNTER

logger = logging.getLogger(__name__)
router = APIRouter()

# to get the details of full pipeline
@router.get("/pipeline/status", response_model=PipelineStatusResponse)
def pipeline_status() -> PipelineStatusResponse:
    drift_result = DriftService.get_latest_drift_status()
    return PipelineStatusResponse(
        last_ingestion=drift_result.get("last_check", "N/A"),
        drift_detected=drift_result.get("drift_detected", False),
        drift_score=drift_result.get("max_drift_score", 0.0),
        model_version=ModelService.get_version(),
        total_predictions=int(PREDICTION_COUNTER._value.get()),
    )
