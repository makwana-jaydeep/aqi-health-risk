import os
import logging
from typing import Any, Dict, Optional

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

INDIAN_CITIES = [
    "Ahmedabad", "Bengaluru", "Bhopal", "Chennai", "Delhi",
    "Hyderabad", "Jaipur", "Kanpur", "Kolkata", "Lucknow",
    "Mumbai", "Nagpur", "Patna", "Pune", "Surat",
]

RISK_COLORS = {
    "Safe": "#27AE60",
    "Caution": "#F39C12",
    "Avoid Outdoors": "#E74C3C",
}

RISK_ICONS = {
    "Safe": "CHECK",
    "Caution": "WARNING",
    "Avoid Outdoors": "STOP",
}

st.set_page_config(
    page_title="AQI Health Risk Assessment",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-card {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
    }
    .risk-safe { background-color: #d5f5e3; border: 2px solid #27AE60; }
    .risk-caution { background-color: #fef9e7; border: 2px solid #F39C12; }
    .risk-avoid { background-color: #fdedec; border: 2px solid #E74C3C; }
    .metric-box {
        background-color: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #3498DB;
        margin: 8px 0;
    }
    .sidebar-section {
        background-color: #ecf0f1;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def call_predict(payload: Dict[str, Any]) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend API. Ensure the backend service is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend may be under load.")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} - {exc.response.text}")
        return None


def call_health() -> Optional[Dict]:
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def call_pipeline_status() -> Optional[Dict]:
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/pipeline/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def render_risk_assessment_page():
    st.title("Personalized Air Quality Health Risk Assessment")
    st.markdown(
        "Enter your city, current pollution readings, and health profile "
        "to receive a personalised health risk recommendation."
    )

    with st.sidebar:
        st.header("Health Profile")

        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        age_group = st.selectbox(
            "Age Group",
            options=["child (0-12)", "teen (13-17)", "adult (18-59)", "senior (60+)"],
            index=2,
        )
        age_group_value = age_group.split(" ")[0]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        has_asthma = st.checkbox("I have asthma or respiratory conditions")
        has_heart = st.checkbox("I have a heart condition")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        planned_activity = st.radio(
            "Planned outdoor activity",
            options=["low", "moderate", "high"],
            index=1,
            help="low = sitting/walking slowly, moderate = brisk walking/cycling, high = jogging/sports",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Location")
        city = st.selectbox("Select your city", options=INDIAN_CITIES, index=4)

        st.subheader("Current Pollution Readings")
        aqi = st.slider("AQI (Air Quality Index)", min_value=0, max_value=500, value=150,
                        help="Overall AQI value published by CPCB")
        pm25 = st.number_input("PM2.5 (ug/m3)", min_value=0.0, max_value=400.0, value=90.0, step=1.0)
        pm10 = st.number_input("PM10 (ug/m3)", min_value=0.0, max_value=500.0, value=180.0, step=1.0)
        no2 = st.number_input("NO2 (ug/m3)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)

    with col2:
        st.subheader("Current Weather")
        temperature = st.slider("Temperature (C)", min_value=0, max_value=50, value=30)
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
        wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, value=10)

        st.subheader("AQI Gauge")
        fig = _aqi_gauge(aqi)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    if st.button("Assess My Risk", type="primary", use_container_width=True):
        payload = {
            "city": city,
            "aqi": float(aqi),
            "pm25": float(pm25),
            "pm10": float(pm10),
            "no2": float(no2),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "wind_speed": float(wind_speed),
            "age_group": age_group_value,
            "has_asthma": has_asthma,
            "has_heart_condition": has_heart,
            "planned_activity": planned_activity,
        }

        with st.spinner("Assessing risk..."):
            result = call_predict(payload)

        if result:
            _render_result(result)


def _render_result(result: Dict[str, Any]):
    tier = result["risk_tier"]
    confidence = result["confidence"]
    recommendation = result["recommendation"]
    city = result["city"]

    css_class = {
        "Safe": "risk-safe",
        "Caution": "risk-caution",
        "Avoid Outdoors": "risk-avoid",
    }.get(tier, "risk-safe")

    color = RISK_COLORS.get(tier, "#27AE60")

    st.markdown(f"""
    <div class='risk-card {css_class}'>
        <h2 style='color: {color}; margin: 0;'>{tier}</h2>
        <p style='font-size: 1.1rem; color: #555; margin: 8px 0;'>Risk level for {city}</p>
        <p style='font-size: 0.9rem; color: #888;'>Model confidence: {confidence:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"Recommendation: {recommendation}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Risk Level", tier)
    with col_b:
        st.metric("Confidence", f"{confidence:.0%}")
    with col_c:
        st.metric("Model Version", result.get("model_version", "N/A"))


def _aqi_gauge(aqi: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={"text": "AQI", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": "#2c3e50"},
            "steps": [
                {"range": [0, 50], "color": "#27AE60"},
                {"range": [50, 100], "color": "#F1C40F"},
                {"range": [100, 200], "color": "#E67E22"},
                {"range": [200, 300], "color": "#E74C3C"},
                {"range": [300, 500], "color": "#8E44AD"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": aqi,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render_pipeline_monitor_page():
    st.title("ML Pipeline Monitor")
    st.markdown("Real-time status of the data ingestion and model training pipeline.")

    health = call_health()
    if health:
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "green" if health.get("status") == "ok" else "red"
            st.markdown(
                f"<div class='metric-box'><b>API Status</b><br>"
                f"<span style='color:{status_color};font-size:1.2rem;'>{health.get('status','unknown').upper()}</span></div>",
                unsafe_allow_html=True,
            )
        with col2:
            model_loaded = health.get("model_loaded", False)
            st.markdown(
                f"<div class='metric-box'><b>Model Loaded</b><br>"
                f"<span style='font-size:1.2rem;'>{'Yes' if model_loaded else 'No'}</span></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='metric-box'><b>API Version</b><br>"
                f"<span style='font-size:1.2rem;'>{health.get('version','N/A')}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Backend API is not reachable. Start the backend service.")

    st.markdown("---")

    pipeline_status = call_pipeline_status()
    if pipeline_status:
        st.subheader("Pipeline Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", pipeline_status.get("total_predictions", 0))
        with col2:
            drift = pipeline_status.get("drift_detected", False)
            st.metric("Drift Detected", "Yes" if drift else "No", delta=None)
        with col3:
            st.metric("Drift Score", f"{pipeline_status.get('drift_score', 0.0):.4f}")
        with col4:
            st.metric("Model Version", pipeline_status.get("model_version", "N/A"))

        st.markdown(f"**Last ingestion check:** {pipeline_status.get('last_ingestion', 'N/A')}")

    st.markdown("---")
    st.subheader("External Dashboards")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "[Open Airflow Pipeline](http://localhost:8080) - View DAG runs and task logs"
        )
    with col2:
        st.markdown(
            "[Open MLflow Experiments](http://localhost:5000) - View training runs and models"
        )
    with col3:
        st.markdown(
            "[Open Grafana Dashboard](http://localhost:3000) - View live metrics and alerts"
        )

    st.subheader("AQI Standard Reference")
    aqi_standards = pd.DataFrame({
        "AQI Range": ["0 - 50", "51 - 100", "101 - 200", "201 - 300", "301 - 400", "401 - 500"],
        "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "Health Impact": [
            "Minimal impact",
            "Minor discomfort to sensitive individuals",
            "Discomfort to people with lung/heart diseases",
            "Breathing discomfort on prolonged outdoor exposure",
            "Respiratory illness on prolonged exposure",
            "Health impacts even on light activity",
        ],
    })
    st.dataframe(aqi_standards, use_container_width=True, hide_index=True)


def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Risk Assessment", "Pipeline Monitor"],
        index=0,
    )

    if page == "Risk Assessment":
        render_risk_assessment_page()
    else:
        render_pipeline_monitor_page()

    st.sidebar.markdown("---")
    st.sidebar.caption("AQI Health Risk v1.0.0 | Data: CPCB India")


if __name__ == "__main__":
    main()
