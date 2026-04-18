import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.drift_detection import compute_baseline_stats, detect_drift

import pandas as pd


def make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "aqi": rng.normal(150, 50, n),
        "pm25": rng.normal(90, 30, n),
        "pm10": rng.normal(180, 60, n),
        "no2": rng.normal(40, 15, n),
        "temperature": rng.normal(30, 5, n),
        "humidity": rng.normal(60, 15, n),
        "wind_speed": rng.exponential(10, n),
    })


def test_compute_baseline_stats_creates_file():
    df = make_df()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        compute_baseline_stats(df, path)
        assert os.path.exists(path)
        with open(path) as f:
            stats = json.load(f)
        assert "aqi" in stats
        assert "mean" in stats["aqi"]
        assert "samples" in stats["aqi"]
    finally:
        os.unlink(path)


def test_detect_drift_no_drift_same_distribution():
    df = make_df(seed=42)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        compute_baseline_stats(df, path)
        current = {col: df[col].tolist() for col in df.columns}
        result = detect_drift(path, current)
        assert not result["drift_detected"]
    finally:
        os.unlink(path)


def test_detect_drift_detects_large_shift():
    df_baseline = make_df(seed=42)
    rng = np.random.default_rng(99)
    df_current = pd.DataFrame({
        "aqi": rng.normal(350, 50, 500),
        "pm25": rng.normal(220, 30, 500),
        "pm10": rng.normal(380, 60, 500),
        "no2": rng.normal(40, 15, 500),
        "temperature": rng.normal(30, 5, 500),
        "humidity": rng.normal(60, 15, 500),
        "wind_speed": rng.exponential(10, 500),
    })
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        compute_baseline_stats(df_baseline, path)
        current = {col: df_current[col].tolist() for col in df_current.columns}
        result = detect_drift(path, current)
        assert result["drift_detected"]
    finally:
        os.unlink(path)
