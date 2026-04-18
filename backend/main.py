import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.routes import health, pipeline, predict
from core.config import settings
from core.logging_config import setup_logging
from services.model_service import ModelService

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AQI Health Risk API starting")
    ModelService.load_model()
    yield
    logger.info("AQI Health Risk API shutting down")


app = FastAPI(
    title="AQI Personal Health Risk API",
    version="1.0.0",
    description="Personalized air quality health risk assessment for Indian cities",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["pipeline"])
