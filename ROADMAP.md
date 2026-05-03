# Senior-Level Project Roadmap
## California Housing Price Prediction — ML Engineer Edition

**Goal:** Transform this project into a production-grade ML system that any US tech company would accept as a portfolio piece.

**Current level:** Strong Junior → Mid  
**Target level:** Strong Mid → Entry Senior  

---

## What exists right now ✅

```
src/data/          loader.py, preprocessor.py, splitter.py
src/features/      build_features.py, feature_selector.py
src/models/        baseline, regularized, tree_based, gradient_boosting, model_factory
src/evaluation/    metrics.py, comparison.py, diagnostics.py
common/            logger, eval_utils, plot_utils, data_utils, reproducibility
configs/           model_config.yaml, training_config.yaml, data_config.yaml
tests/             test_data_loader.py
```

## What is missing ❌

```
Training pipeline     — no train.py, nothing ties everything together
FastAPI serving       — no API, model can't be called from outside
CI/CD                 — no GitHub Actions, nothing runs on push
Docker                — .dockerignore exists but no Dockerfile
Abstract base class   — models have no common interface
Custom exceptions     — bare RuntimeError/ValueError everywhere
Pydantic settings     — config not validated, no .env support
SHAP explainability   — installed but no code
Evidently monitoring  — installed but no code
Prediction logging    — no database, no audit trail
Pre-commit hooks      — nothing checks code before commit
DVC                   — data not versioned
More tests            — only 1 test file, 0% coverage on models/features/API
Editable install      — sys.path hack causes E402 everywhere
```

---

## PHASE 1 — Fix the Foundation (do this first, everything depends on it)

### Step 1 — Editable install (kills all E402 errors forever)

**Why:** Every `sys.path.insert` in every file causes Ruff E402.  
Editable install makes Python find your packages without hacks.

Add to `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "common*"]
```

Run once:
```bash
uv pip install -e .
```

Then delete `sys.path.insert(...)` from every `.py` file.

**Tools:** `uv`, `setuptools`  
**Done when:** Zero E402 errors. Zero Pylance import squiggles.

---

### Step 2 — Pre-commit hooks (auto-check before every commit)

**Why:** Catches Ruff + Pyright errors before they enter git history.

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/RobertCraigie/pyright-python
    rev: v1.1.350
    hooks:
      - id: pyright
```

Install:
```bash
uv pip install pre-commit
pre-commit install
```

**Tools:** `pre-commit`, `ruff`, `pyright`  
**Done when:** `git commit` auto-runs Ruff + Pyright. Bad code is rejected before commit.

---

### Step 3 — Custom exceptions (replace bare RuntimeError/ValueError)

**Why:** Bare exceptions give no context. Custom ones tell you exactly what failed.

Create `src/exceptions.py`:
```python
class HousingPredictionError(Exception):
    """Base exception for this project."""

class DataValidationError(HousingPredictionError):
    """Raised when input data fails validation."""

class ModelNotFittedError(HousingPredictionError):
    """Raised when transform/predict called before fit."""

class FeatureError(HousingPredictionError):
    """Raised when feature engineering fails."""

class ConfigurationError(HousingPredictionError):
    """Raised when config is invalid or missing required keys."""
```

Then replace across the codebase:
```python
# ❌ Before
raise RuntimeError("Call fit() first.")

# ✅ After
raise ModelNotFittedError(
    f"{type(self).__name__} is not fitted. Call fit(X_train) first."
)
```

**Tools:** Pure Python  
**Done when:** No bare `RuntimeError` / `ValueError` in `src/`.

---

### Step 4 — Pydantic settings (replace raw YAML loading)

**Why:** YAML loaded with `open()` has no validation — wrong types crash at runtime.  
Pydantic catches config errors at startup, not mid-training.

Create `src/config.py`:
```python
from pydantic_settings import BaseSettings
from pydantic import Field

class TrainingConfig(BaseSettings):
    experiment_name: str = "regression_california_housing"
    random_state: int = 42
    n_trials: int = 50
    test_size: float = Field(default=0.2, ge=0.0, lt=1.0)
    val_size: float = Field(default=0.1, ge=0.0, lt=1.0)
    primary_metric: str = "rmse"

    model_config = {"env_prefix": "HOUSING_"}

class AppConfig(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_path: str = "models/saved/best_model.joblib"
    rate_limit: str = "100/minute"

    model_config = {"env_file": ".env"}
```

**Tools:** `pydantic-settings`, `python-dotenv`  
**Done when:** Config loaded from `.env` + YAML with type validation. No raw `yaml.safe_load` in src/.

---

## PHASE 2 — Abstract Base Class + Training Pipeline

### Step 5 — Abstract base class for all models

**Why:** Without a common interface, every model is different.  
With ABC, you guarantee every model has `fit`, `predict`, `get_params`.  
This is how sklearn itself is designed.

Create `src/models/base.py`:
```python
from abc import ABC, abstractmethod
import numpy as np

class BaseRegressor(ABC):
    """Every model in this project must implement this interface."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressor":
        """Fit the model. Must return self."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as 1D array."""
        ...

    @abstractmethod
    def get_params(self) -> dict[str, object]:
        """Return model hyperparameters as a dict."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.get_params()})"
```

**Tools:** `abc` (stdlib)  
**Done when:** Every model class in `src/models/` inherits from `BaseRegressor`.

---

### Step 6 — Training pipeline (the most important missing piece)

**Why:** Right now nothing ties loader → features → preprocess → train → evaluate → save.  
A pipeline is the heart of any ML project.

Create `src/pipeline/train_pipeline.py`:
```python
# What it does, in order:
# 1. Load config (Pydantic)
# 2. Load data (loader.py)
# 3. Validate data (loader.py)
# 4. Build features (build_features.py)
# 5. Split data (splitter.py)
# 6. Preprocess (preprocessor.py)
# 7. For each enabled model in model_config.yaml:
#    a. Tune hyperparameters (Optuna)
#    b. Train best params
#    c. Evaluate on val set
#    d. Log to MLflow
#    e. Store in ModelComparison
# 8. Select best model
# 9. Re-train best model on train+val
# 10. Final evaluation on test set
# 11. Save model (joblib)
# 12. Generate SHAP plot
# 13. Plot comparison chart
# 14. Save results CSV
```

Create `train.py` (entry point at project root):
```bash
python train.py --config configs/training_config.yaml
```

**Tools:** `mlflow`, `optuna`, `joblib`, `click`, `pydantic-settings`  
**Done when:** `python train.py` runs end-to-end and saves a model to `models/saved/`.

---

## PHASE 3 — FastAPI Serving

### Step 7 — FastAPI application

**Why:** A model that can't be called from outside is useless in production.  
FastAPI is the industry standard for ML model serving in Python.

Create `src/api/`:
```
src/api/
├── __init__.py
├── app.py          ← FastAPI app, lifespan, router registration
├── routes/
│   ├── predict.py  ← POST /predict endpoint
│   ├── health.py   ← GET /health, GET /ready
│   └── explain.py  ← GET /explain (SHAP values)
├── schemas/
│   ├── request.py  ← PredictRequest (Pydantic model)
│   └── response.py ← PredictResponse (Pydantic model)
└── middleware/
    └── logging.py  ← log every request/response
```

Endpoints:
```
GET  /health          → {"status": "ok"}
GET  /ready           → {"status": "ready", "model": "xgboost"}
POST /predict         → {"prediction": 3.42, "confidence_interval": [3.1, 3.7]}
GET  /explain?idx=0   → {"shap_values": [...], "feature_names": [...]}
GET  /metrics         → Prometheus metrics endpoint
```

Request validation with Pydantic:
```python
class PredictRequest(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in block")
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32.0, le=42.0)
    Longitude: float = Field(..., ge=-125.0, le=-114.0)
```

**Tools:** `fastapi`, `uvicorn`, `pydantic`, `httpx`  
**Done when:** `uvicorn src.api.app:app` starts. All endpoints respond. Tested with Postman.

---

### Step 8 — Rate limiting + API key auth

**Why:** Without rate limiting anyone can DDoS your API.  
Without auth anyone can call your endpoint.

```python
# Rate limiting with SlowAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    ...

# API key auth
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

**Tools:** `slowapi`, `fastapi.security`, `python-dotenv`  
**Done when:** API returns 429 after 100 req/min. Returns 403 without API key.

---

### Step 9 — Prediction logging

**Why:** In production you must log every prediction for auditing, debugging, and drift detection.

Create `src/api/logging_db.py`:
```python
# Log to SQLite (local) or PostgreSQL (production)
# Each row:
# - timestamp
# - request_id (UUID)
# - input features (JSON)
# - prediction value
# - model version
# - latency_ms
```

**Tools:** `sqlite3` (stdlib) or `sqlalchemy`, `uuid`  
**Done when:** Every `/predict` call writes a row to `predictions.db`.

---

## PHASE 4 — Explainability + Monitoring

### Step 10 — SHAP explainability

**Why:** US companies ask "why did the model predict this?" SHAP answers it.

Create `src/explainability/shap_explainer.py`:
```python
import shap

class ShapExplainer:
    def __init__(self, model, X_train: np.ndarray) -> None:
        self._explainer = shap.TreeExplainer(model)
        self._X_train = X_train

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values for input X."""
        return self._explainer.shap_values(X)

    def plot_summary(self, X: np.ndarray, feature_names: list[str]) -> None:
        shap.summary_plot(self.explain(X), X, feature_names=feature_names)

    def get_feature_importance(self, X: np.ndarray) -> dict[str, float]:
        shap_vals = np.abs(self.explain(X)).mean(axis=0)
        return dict(zip(self._feature_names, shap_vals.tolist()))
```

**Tools:** `shap`  
**Done when:** `/explain` endpoint returns per-feature SHAP values. Summary plot saved to `experiments/figures/`.

---

### Step 11 — Data drift monitoring with Evidently

**Why:** Models degrade when real-world data shifts from training data.  
Evidently detects this automatically.

Create `src/monitoring/drift_detector.py`:
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

class DriftDetector:
    def run_report(
        self,
        reference: pd.DataFrame,  # training data
        current: pd.DataFrame,    # new incoming data
        save_path: str,
    ) -> dict[str, bool]:
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(save_path)
        return report.as_dict()
```

**Tools:** `evidently`  
**Done when:** Script generates HTML drift report comparing training vs recent predictions.

---

## PHASE 5 — Testing (coverage must be ≥ 80%)

### Step 12 — Full test suite

Create these test files:

```
tests/
├── test_data_loader.py        ← exists ✅
├── test_build_features.py     ← missing ❌
├── test_preprocessor.py       ← missing ❌
├── test_feature_selector.py   ← missing ❌
├── test_model_factory.py      ← missing ❌
├── test_metrics.py            ← missing ❌
├── test_comparison.py         ← missing ❌
├── test_api.py                ← missing ❌ (FastAPI integration tests)
└── conftest.py                ← missing ❌ (shared fixtures)
```

`conftest.py` holds shared fixtures:
```python
import pytest
import numpy as np
import pandas as pd

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Reused across all test files."""
    ...

@pytest.fixture(scope="session")
def fitted_preprocessor(sample_df):
    ...

@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from src.api.app import app
    return TestClient(app)
```

Run coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Tools:** `pytest`, `pytest-cov`, `pytest-asyncio`, `httpx`  
**Done when:** Coverage ≥ 80% across all `src/` files.

---

## PHASE 6 — CI/CD with GitHub Actions

### Step 13 — CI pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv pip install -e ".[dev]"
      - run: ruff check .
      - run: pyright

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=src --cov-fail-under=80

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: false
          tags: housing-prediction:latest
```

**Tools:** GitHub Actions, `uv`, `ruff`, `pyright`, `pytest`, `docker`  
**Done when:** Every `git push` triggers lint + test + docker build. PR blocked if any fails.

---

## PHASE 7 — Docker + Deployment

### Step 14 — Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install uv && uv pip install --system -r requirements.txt

COPY . .
RUN uv pip install --system -e .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HOUSING_MODEL_PATH=models/saved/best_model.joblib
    volumes:
      - ./models:/app/models
```

**Tools:** Docker, docker-compose  
**Done when:** `docker compose up` starts the API. `curl localhost:8000/health` returns 200.

---

### Step 15 — Load testing

Create `tests/load_test.py` (Locust):
```python
from locust import HttpUser, task, between

class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)
    headers = {"X-API-Key": "test-key"}

    @task
    def predict(self):
        self.client.post("/predict", json={
            "MedInc": 3.5, "HouseAge": 20.0,
            "AveRooms": 5.0, "AveBedrms": 1.0,
            "Population": 800.0, "AveOccup": 3.0,
            "Latitude": 37.5, "Longitude": -122.0,
        }, headers=self.headers)
```

Run:
```bash
locust -f tests/load_test.py --host=http://localhost:8000
```

**Tools:** `locust`  
**Done when:** API handles 100 concurrent users with < 200ms p95 latency.

---

## PHASE 8 — Data Versioning + Model Card

### Step 16 — DVC for data versioning

**Why:** You must track which data version produced which model. DVC does this.

```bash
dvc init
dvc add data/raw/california_housing.csv
git add data/raw/california_housing.csv.dvc .gitignore
git commit -m "track raw data with DVC"
```

**Tools:** `dvc`  
**Done when:** `dvc repro` re-runs the full pipeline from data → model.

---

### Step 17 — Model card

Create `MODEL_CARD.md`:
```markdown
# Model Card — California Housing Price Predictor

## Model details
- Type: XGBoost Regressor (best from 11 models compared)
- Version: 1.0.0
- Trained: 2026-04-23

## Performance
| Metric | Train | Val  | Test |
|--------|-------|------|------|
| RMSE   | 0.41  | 0.45 | 0.46 |
| MAE    | 0.29  | 0.31 | 0.32 |
| R²     | 0.87  | 0.85 | 0.84 |

## Intended use
Predict median house value for California census blocks.

## Limitations
- Trained on 1990 census data — not suitable for current prices
- May underperform in data-sparse regions

## Ethical considerations
- Do not use for individual loan decisions
- Geographic features (Lat/Long) may encode demographic bias
```

**Tools:** Markdown  
**Done when:** `MODEL_CARD.md` exists at project root with real numbers filled in.

---

## Full tools and technologies map

### Development
| Tool | Purpose |
|------|---------|
| `uv` | Virtual environment + package management |
| `ruff` | Linter + formatter + import sorter |
| `pyright` | Static type checker |
| `pre-commit` | Auto-run checks before every commit |

### Data & ML
| Tool | Purpose |
|------|---------|
| `numpy` | Arrays and math |
| `pandas` | DataFrames |
| `scikit-learn` | Models, preprocessing, metrics |
| `xgboost` | Best performing model |
| `lightgbm` | Fast gradient boosting |
| `catboost` | Handles categories natively |
| `optuna` | Hyperparameter tuning |
| `joblib` | Save/load models |

### Experiment tracking
| Tool | Purpose |
|------|---------|
| `mlflow` | Log metrics, params, artifacts per run |

### Explainability + Monitoring
| Tool | Purpose |
|------|---------|
| `shap` | Explain individual predictions |
| `evidently` | Detect data drift over time |

### Serving
| Tool | Purpose |
|------|---------|
| `fastapi` | REST API |
| `uvicorn` | ASGI server that runs FastAPI |
| `pydantic` | Request/response validation |
| `slowapi` | Rate limiting |

### Configuration
| Tool | Purpose |
|------|---------|
| `pydantic-settings` | Type-safe config from `.env` + YAML |
| `python-dotenv` | Load `.env` files |
| `pyyaml` | Read YAML config files |

### Testing
| Tool | Purpose |
|------|---------|
| `pytest` | Run tests |
| `pytest-cov` | Measure coverage |
| `pytest-asyncio` | Test async FastAPI routes |
| `httpx` | HTTP client for API tests |
| `locust` | Load testing |

### Infrastructure
| Tool | Purpose |
|------|---------|
| `docker` | Package app + dependencies |
| `docker-compose` | Run multi-service stack locally |
| GitHub Actions | CI/CD — lint + test + build on every push |
| `dvc` | Data and pipeline versioning |

### Visualization
| Tool | Purpose |
|------|---------|
| `matplotlib` | Base plots |
| `seaborn` | Statistical plots |
| `plotly` | Interactive charts |

---

## Progress tracker

| Phase | Step | Status |
|-------|------|--------|
| 1 | Editable install | ❌ |
| 1 | Pre-commit hooks | ❌ |
| 1 | Custom exceptions | ❌ |
| 1 | Pydantic settings | ❌ |
| 2 | Abstract base class | ❌ |
| 2 | Training pipeline | ❌ |
| 3 | FastAPI app | ❌ |
| 3 | Rate limit + auth | ❌ |
| 3 | Prediction logging | ❌ |
| 4 | SHAP explainability | ❌ |
| 4 | Evidently monitoring | ❌ |
| 5 | Full test suite ≥80% | ❌ |
| 6 | GitHub Actions CI | ❌ |
| 7 | Dockerfile | ❌ |
| 7 | Load test | ❌ |
| 8 | DVC | ❌ |
| 8 | Model card | ❌ |

Mark each ❌ → ✅ as you complete it.

---

## What this project looks like when done

```
house_prediction/
├── .github/workflows/ci.yml      ← CI/CD
├── .pre-commit-config.yaml       ← auto code checks
├── .env.example                  ← config template
├── Dockerfile                    ← containerization
├── docker-compose.yml            ← local stack
├── pyproject.toml                ← editable install + ruff + pyright
├── train.py                      ← one command trains everything
├── MODEL_CARD.md                 ← documents model performance
├── ROADMAP.md                    ← this file
├── CODE_STYLE.md                 ← coding standards
├── common/                       ← shared utilities
├── configs/                      ← yaml configs
├── data/raw/                     ← DVC tracked
├── experiments/                  ← MLflow runs, figures, results
├── models/saved/                 ← trained model artifacts
├── src/
│   ├── api/                      ← FastAPI app
│   ├── config.py                 ← Pydantic settings
│   ├── exceptions.py             ← custom exceptions
│   ├── data/                     ← loader, preprocessor, splitter
│   ├── features/                 ← feature engineering
│   ├── models/                   ← base class + all models
│   ├── evaluation/               ← metrics, comparison, diagnostics
│   ├── explainability/           ← SHAP
│   ├── monitoring/               ← Evidently drift
│   └── pipeline/                 ← training pipeline
└── tests/                        ← full test suite ≥80% coverage
```

**Every piece of this exists in production ML systems at companies like Airbnb, Spotify, and Uber.**
