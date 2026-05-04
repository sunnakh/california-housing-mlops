# California Housing Price Predictor

A production-grade ML system that predicts California house prices. Built with real MLOps practices — not just a notebook.

## What's inside

- **Training pipeline** — loads data, engineers features, trains multiple models, picks the best one
- **Hyperparameter tuning** — Optuna Bayesian optimization across XGBoost, LightGBM, and RandomForest
- **Experiment tracking** — every run, metric, and model version logged to MLflow
- **REST API** — FastAPI serving with `/predict`, `/predict/batch`, and `/health` endpoints
- **Containerized** — Docker image with non-root user, healthcheck, and layer caching
- **CI/CD** — GitHub Actions runs lint, tests, and Docker build on every push

## Results

| Model | RMSE | R² |
|---|---|---|
| XGBoost (tuned) | 0.4616 | 0.848 |
| LightGBM (tuned) | 0.4636 | 0.847 |
| RandomForest (tuned) | 0.5181 | 0.808 |

## Quickstart

```bash
# Install
uv sync && uv pip install -e .

# Tune hyperparameters
.venv/bin/python pipeline/tune.py --config configs/tuning_config.yaml --training-config configs/training_config.yaml

# Train
.venv/bin/python pipeline/train.py --config configs/training_config.yaml

# Serve
uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 3.5, "HouseAge": 28.0, "AveRooms": 5.2, "AveBedrms": 1.1, "Population": 1200.0, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23}'
```

## MLflow UI

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` to compare experiments and model versions.

## Docker

```bash
docker build -t house-prediction-api .
docker run -p 8000:8000 house-prediction-api
```

## Stack

Python 3.12 · XGBoost · LightGBM · scikit-learn · FastAPI · MLflow · Optuna · Docker · GitHub Actions
