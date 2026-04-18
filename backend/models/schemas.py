from enum import Enum
from pydantic import BaseModel, Field


class AgeGroup(str, Enum):
    child = "child"
    teen = "teen"
    adult = "adult"
    senior = "senior"


class ActivityLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"


class RiskTier(str, Enum):
    safe = "Safe"
    caution = "Caution"
    avoid = "Avoid Outdoors"


class PredictionRequest(BaseModel):
    city: str = Field(..., description="Indian city name", json_schema_extra={"example": "Delhi"})
    aqi: float = Field(..., ge=0, le=500, description="Overall AQI value", json_schema_extra={"example": 185.0})
    pm25: float = Field(..., ge=0, description="PM2.5 concentration in ug/m3", json_schema_extra={"example": 120.5})
    pm10: float = Field(..., ge=0, description="PM10 concentration in ug/m3", json_schema_extra={"example": 210.0})
    no2: float = Field(..., ge=0, description="NO2 concentration in ug/m3", json_schema_extra={"example": 45.0})
    temperature: float = Field(..., description="Temperature in Celsius", json_schema_extra={"example": 32.0})
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in percent", json_schema_extra={"example": 65.0})
    wind_speed: float = Field(..., ge=0, description="Wind speed in km/h", json_schema_extra={"example": 8.0})
    age_group: AgeGroup = Field(..., description="User age group", json_schema_extra={"example": "adult"})
    has_asthma: bool = Field(..., description="Whether the user has asthma", json_schema_extra={"example": False})
    has_heart_condition: bool = Field(..., description="Whether the user has a heart condition", json_schema_extra={"example": False})
    planned_activity: ActivityLevel = Field(..., description="Planned outdoor activity level", json_schema_extra={"example": "moderate"})


class PredictionResponse(BaseModel):
    city: str
    risk_tier: RiskTier
    risk_score: int = Field(..., ge=0, le=2, description="Numeric risk score: 0=Safe, 1=Caution, 2=Avoid Outdoors")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    baseline_loaded: bool


class PipelineStatusResponse(BaseModel):
    last_ingestion: str
    drift_detected: bool
    drift_score: float
    model_version: str
    total_predictions: int
