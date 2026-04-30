# Project Commands

## Setup

```bash
# Install dependencies
uv sync

# Install project as editable package
uv pip install -e .
```

---

## Running Tests

```bash
# Run all tests
uv run --with pytest python -m pytest tests/ -v

# Run a single test file
uv run --with pytest python -m pytest tests/test_comparison.py -v

# Run a single test
uv run --with pytest python -m pytest tests/test_comparison.py::TestComparison::test_add_results -v

# Run with coverage report
uv run --with pytest python -m pytest --cov=src --cov-report=term-missing -q

# Run with coverage — show only files below 80%
uv run --with pytest python -m pytest --cov=src --cov-report=term-missing -q 2>/dev/null | grep -E "([0-6][0-9]%|7[0-9]%|TOTAL|FAIL)"
```

---

## Linting and Type Checking

```bash
# Lint
uv run --no-project ruff check .

# Lint and auto-fix
uv run --no-project ruff check . --fix

# Type check
uv run --no-project pyright
```

---

## Training Pipeline

```bash
# Run full training pipeline
python pipelines/run_training.py --config configs/training_config.yaml
```

---

## Serving the API

```bash
# Start FastAPI server
uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000 --reload

# Or via pipeline runner
python pipelines/run_serve.py --config configs/deployment_config.yaml
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Liveness check |
| POST | /predict | Single prediction |
| POST | /predict/batch | Batch prediction (max 1000) |
| GET | /docs | Swagger UI |
| GET | /redoc | ReDoc documentation |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 28.0,
    "AveRooms": 5.2,
    "AveBedrms": 1.1,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

---

## Coming Soon

- Docker commands (build, run, compose)
- MLflow experiment tracking
- Optuna hyperparameter tuning
