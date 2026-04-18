import logging
import time

from fastapi import APIRouter, HTTPException

from models.schemas import PredictionRequest, PredictionResponse
from services.model_service import ModelService
from services.prometheus_service import (
    PREDICTION_COUNTER,
    PREDICTION_DURATION,
    RISK_TIER_COUNTER,
)

logger = logging.getLogger(__name__)
router = APIRouter()

RECOMMENDATIONS = {
    "Safe": "Air quality is acceptable. Outdoor activities can proceed normally.",
    "Caution": "Sensitive individuals should limit prolonged outdoor exertion. "
               "Consider wearing an N95 mask for extended outdoor stays.",
    "Avoid Outdoors": "Air quality is hazardous. Avoid all outdoor activities. "
                     "Keep windows closed and use an air purifier if available.",
}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    logger.info("Prediction request for city=%s aqi=%.1f", request.city, request.aqi)

    start = time.time()
    try:
        result = ModelService.predict(request)
    except RuntimeError as exc:
        logger.error("Model inference failed: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available") from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    duration = time.time() - start
    PREDICTION_COUNTER.inc()
    PREDICTION_DURATION.observe(duration)
    RISK_TIER_COUNTER.labels(tier=result.risk_tier).inc()

    logger.info(
        "Prediction complete city=%s risk=%s confidence=%.3f duration=%.3fs",
        request.city,
        result.risk_tier,
        result.confidence,
        duration,
    )

    result.recommendation = RECOMMENDATIONS[result.risk_tier]
    return result
