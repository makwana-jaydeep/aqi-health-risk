import logging
import time
import json
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pathlib import Path
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

# to do prediction
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

# check feedback
@router.post("/feedback")
def submit_feedback(city: str, actual_risk: int, predicted_risk: int):
    if actual_risk not in (0, 1, 2) or predicted_risk not in (0, 1, 2):
        raise HTTPException(status_code=422, detail="Risk values must be 0, 1, or 2")

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "city": city,
        "actual_risk": actual_risk,
        "predicted_risk": predicted_risk,
        "correct": actual_risk == predicted_risk,
    }


    DATA_DIR = os.getenv("DATA_DIR", "data")  # fallback for local/testing
    feedback_path = Path(DATA_DIR) / "feedback_log.jsonl"

    try:
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        with open(feedback_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info("Feedback logged: predicted=%d actual=%d", predicted_risk, actual_risk)

    except Exception as exc:
        logger.error("Failed to write feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to log feedback") from exc

    return {"status": "logged"}
