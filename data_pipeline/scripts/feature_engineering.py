import logging

import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "aqi" in df.columns and "pm25" in df.columns:
        df["pm25_aqi_ratio"] = (df["pm25"] / (df["aqi"] + 1)).round(4)

    if "humidity" in df.columns and "temperature" in df.columns:
        df["heat_index"] = (df["temperature"] + 0.33 * (df["humidity"] / 100 * 6.105) - 4.0).round(2)

    if "wind_speed" in df.columns:
        df["low_wind"] = (df["wind_speed"] < 5).astype(int)

    if "aqi" in df.columns:
        df["aqi_severity"] = pd.cut(
            df["aqi"],
            bins=[0, 50, 100, 200, 300, 500],
            labels=[0, 1, 2, 3, 4],
            right=True,
        ).cat.add_categories([-1]).fillna(-1).astype(int)

    logger.info("Feature engineering added columns: pm25_aqi_ratio, heat_index, low_wind, aqi_severity")
    return df
