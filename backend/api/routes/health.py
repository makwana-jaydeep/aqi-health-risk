import logging

from fastapi import APIRouter

from core.config import settings
from models.schemas import HealthResponse, ReadyResponse
from services.drift_service import DriftService
from services.model_service import ModelService

logger = logging.getLogger(__name__)
router = APIRouter()

# to get health advice
@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=ModelService.is_loaded(),
        version=settings.app_version,
    )

# to get advice according to aqi
@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    model_loaded = ModelService.is_loaded()
    baseline_loaded = DriftService.is_baseline_loaded()
    return ReadyResponse(
        ready=model_loaded and baseline_loaded,
        model_loaded=model_loaded,
        baseline_loaded=baseline_loaded,
    )
