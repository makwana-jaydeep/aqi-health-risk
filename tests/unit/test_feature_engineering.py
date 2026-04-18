import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_pipeline.scripts.feature_engineering import engineer_features


def make_df(**kwargs) -> pd.DataFrame:
    defaults = dict(
        aqi=[150.0, 80.0, 30.0],
        pm25=[90.0, 50.0, 20.0],
        pm10=[180.0, 90.0, 40.0],
        no2=[40.0, 30.0, 20.0],
        temperature=[32.0, 25.0, 20.0],
        humidity=[65.0, 55.0, 45.0],
        wind_speed=[8.0, 12.0, 20.0],
    )
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_engineer_features_adds_pm25_aqi_ratio():
    df = make_df()
    result = engineer_features(df)
    assert "pm25_aqi_ratio" in result.columns
    assert (result["pm25_aqi_ratio"] >= 0).all()


def test_engineer_features_adds_heat_index():
    df = make_df()
    result = engineer_features(df)
    assert "heat_index" in result.columns


def test_engineer_features_adds_low_wind():
    df = make_df(wind_speed=[3.0, 10.0, 25.0])
    result = engineer_features(df)
    assert "low_wind" in result.columns
    assert result.iloc[0]["low_wind"] == 1
    assert result.iloc[1]["low_wind"] == 0


def test_engineer_features_adds_aqi_severity():
    df = make_df(
        aqi=[25.0, 75.0, 150.0],
        pm25=[15.0, 45.0, 90.0],
        pm10=[30.0, 90.0, 180.0],
    )
    result = engineer_features(df)
    assert "aqi_severity" in result.columns


def test_engineer_features_does_not_modify_original():
    df = make_df()
    original_cols = set(df.columns)
    engineer_features(df)
    assert set(df.columns) == original_cols


def test_engineer_features_handles_zero_aqi():
    df = make_df(aqi=[0.0, 0.0, 0.0], pm25=[0.0, 0.0, 0.0], pm10=[0.0, 0.0, 0.0])
    result = engineer_features(df)
    assert not result["pm25_aqi_ratio"].isnull().any()
