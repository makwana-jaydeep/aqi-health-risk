#!/bin/bash
set -e

echo "Initializing AQI Health Risk project..."

mkdir -p data/raw data/processed mlruns mlartifacts

if ! command -v git &> /dev/null; then
    echo "git not found. Please install git."
    exit 1
fi

if ! command -v dvc &> /dev/null; then
    echo "dvc not found. Installing..."
    pip install dvc
fi

git init
dvc init

echo "Running DVC pipeline..."
dvc repro

echo "Setup complete."
echo "Start services: docker compose up --build"
