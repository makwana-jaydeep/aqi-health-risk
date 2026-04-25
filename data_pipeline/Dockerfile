FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    apache-airflow[postgres]==2.9.0 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.11.txt"

RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    mlflow==2.11.1 \
    scipy==1.13.0 \
    joblib==1.4.2 \
    requests==2.31.0 \
    pyyaml==6.0.1

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR /opt/airflow

EXPOSE 8080
