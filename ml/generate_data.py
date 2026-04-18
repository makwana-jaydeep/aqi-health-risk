import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def _base_risk(aqi: float) -> int:
    if aqi <= 50:
        return 0
    if aqi <= 150:
        return 1
    return 2


def _personal_modifier(age_group: int, asthma: int, heart: int, activity: int, aqi: float) -> int:
    modifier = 0
    if age_group in (0, 3):
        modifier += 1
    if asthma:
        if aqi > 100:
            modifier += 1
        else:
            modifier += 0
    if heart:
        if aqi > 80:
            modifier += 1
    if activity == 2 and aqi > 80:
        modifier += 1
    return modifier


def generate_dataset(n_samples: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    records = []

    city_profiles = {
        "Delhi": {"aqi_mean": 200, "aqi_std": 80},
        "Mumbai": {"aqi_mean": 130, "aqi_std": 50},
        "Chennai": {"aqi_mean": 100, "aqi_std": 40},
        "Bengaluru": {"aqi_mean": 90, "aqi_std": 35},
        "Kolkata": {"aqi_mean": 160, "aqi_std": 60},
        "Hyderabad": {"aqi_mean": 110, "aqi_std": 45},
        "Pune": {"aqi_mean": 105, "aqi_std": 40},
        "Ahmedabad": {"aqi_mean": 145, "aqi_std": 55},
    }
    city_names = list(city_profiles.keys())

    for _ in range(n_samples):
        city = city_names[rng.integers(0, len(city_names))]
        profile = city_profiles[city]

        aqi = float(np.clip(rng.normal(profile["aqi_mean"], profile["aqi_std"]), 5, 500))
        pm25 = float(np.clip(aqi * rng.uniform(0.55, 0.75) + rng.normal(0, 5), 0, 400))
        pm10 = float(np.clip(aqi * rng.uniform(0.9, 1.3) + rng.normal(0, 10), 0, 500))
        no2 = float(np.clip(rng.normal(30, 15), 0, 200))
        temperature = float(rng.uniform(15, 45))
        humidity = float(rng.uniform(20, 95))
        wind_speed = float(np.clip(rng.exponential(10), 0, 60))
        age_group = int(rng.integers(0, 4))
        has_asthma = int(rng.random() < 0.12)
        has_heart = int(rng.random() < 0.08)
        activity = int(rng.integers(0, 3))

        base = _base_risk(aqi)
        mod = _personal_modifier(age_group, has_asthma, has_heart, activity, aqi)
        risk_score = min(base + min(mod, 1), 2)

        if rng.random() < 0.03:
            risk_score = rng.integers(0, 3)

        records.append({
            "city": city,
            "aqi": round(aqi, 2),
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "no2": round(no2, 2),
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "wind_speed": round(wind_speed, 2),
            "age_group_enc": age_group,
            "has_asthma": has_asthma,
            "has_heart_condition": has_heart,
            "planned_activity_enc": activity,
            "risk_tier": risk_score,
        })

    return pd.DataFrame(records)


def main() -> None:
    params = load_params()
    raw_path = params["data"]["raw_path"]
    n_samples = params["data"]["n_samples"]

    os.makedirs(raw_path, exist_ok=True)
    output_path = os.path.join(raw_path, "aqi_dataset.csv")

    logger.info("Generating %d samples...", n_samples)
    df = generate_dataset(n_samples)
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved to %s shape=%s", output_path, df.shape)

    label_counts = df["risk_tier"].value_counts().to_dict()
    logger.info("Label distribution: %s", label_counts)


if __name__ == "__main__":
    main()
