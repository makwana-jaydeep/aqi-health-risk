import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CPCB_API_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_API_KEY = os.getenv("CPCB_API_KEY", "")

CITY_PROFILES = {
    "Delhi": {"aqi_mean": 200, "aqi_std": 80},
    "Mumbai": {"aqi_mean": 130, "aqi_std": 50},
    "Chennai": {"aqi_mean": 100, "aqi_std": 40},
    "Bengaluru": {"aqi_mean": 90, "aqi_std": 35},
    "Kolkata": {"aqi_mean": 160, "aqi_std": 60},
    "Hyderabad": {"aqi_mean": 110, "aqi_std": 45},
    "Pune": {"aqi_mean": 105, "aqi_std": 40},
    "Ahmedabad": {"aqi_mean": 145, "aqi_std": 55},
}


def fetch_cpcb_data(limit: int = 100) -> Optional[pd.DataFrame]:
    if not CPCB_API_KEY:
        logger.warning("CPCB_API_KEY not set. Using synthetic data fallback.")
        return None

    params = {
        "api-key": CPCB_API_KEY,
        "format": "json",
        "limit": limit,
    }
    try:
        response = requests.get(CPCB_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        records = data.get("records", [])
        if not records:
            logger.warning("CPCB API returned empty records")
            return None
        df = pd.DataFrame(records)
        logger.info("Fetched %d records from CPCB API", len(df))
        return df
    except requests.RequestException as exc:
        logger.error("CPCB API request failed: %s", exc)
        return None


def generate_synthetic_hourly(n_records: int = 200, seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed or int(time.time()))
    records = []
    cities = list(CITY_PROFILES.keys())

    for _ in range(n_records):
        city = cities[rng.integers(0, len(cities))]
        profile = CITY_PROFILES[city]

        hour = rng.integers(0, 24)
        hour_factor = 1.0 + 0.3 * np.sin(np.pi * (hour - 6) / 12)

        aqi = float(np.clip(
            rng.normal(profile["aqi_mean"] * hour_factor, profile["aqi_std"]), 5, 500
        ))
        pm25 = float(np.clip(aqi * rng.uniform(0.55, 0.75), 0, 400))
        pm10 = float(np.clip(aqi * rng.uniform(0.9, 1.3), 0, 500))
        no2 = float(np.clip(rng.normal(30, 15), 0, 200))
        temperature = float(rng.uniform(15, 45))
        humidity = float(rng.uniform(20, 95))
        wind_speed = float(np.clip(rng.exponential(10), 0, 60))

        records.append({
            "city": city,
            "timestamp": datetime.utcnow().isoformat(),
            "aqi": round(aqi, 2),
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "no2": round(no2, 2),
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "wind_speed": round(wind_speed, 2),
        })

    return pd.DataFrame(records)


def ingest(output_dir: str = "/data/raw") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H")
    output_path = os.path.join(output_dir, f"cpcb_{timestamp}.csv")

    df = fetch_cpcb_data()
    if df is None:
        df = generate_synthetic_hourly()

    df.to_csv(output_path, index=False)
    logger.info("Ingested %d records to %s", len(df), output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = ingest()
    print(f"Ingestion complete: {path}")
