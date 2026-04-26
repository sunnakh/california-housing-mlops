"""
FastAPI application for serving the California Housing regression model.

Endpoints:
    GET  /health                — liveness/readiness check
    POST /predict               — single sample prediction
    POST /predict/batch         — batch prediction (up to 1000 samples)
    GET  /docs                  — auto-generated Swagger UI
    GET  /redoc                 — ReDoc documentation

Usage:
    # From the project root (house_prediction/)
    uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000 --reload

    # Or via the CLI wrapper:
    python pipelines/run_serve.py --config configs/deployment_config.yaml
"""

import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logger import get_logger
from src.deployment.model_server import ModelServer
from src.deployment.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HousingFeatures,
    PredictionResponse,
)

logger = get_logger(__name__)

# ── Global model server instance ──────────────────────────────────────────────
# Resolved from deployment_config.yaml at startup.
_model_server: ModelServer | None = None


def _load_deployment_config(config_path: str = "configs/deployment_config.yaml") -> dict:
    """Load deployment configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        # Try relative to the project root
        path = Path(__file__).resolve().parents[2] / config_path
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    FastAPI lifespan context manager.

    Runs at startup: loads the model and preprocessor into memory.
    Runs at shutdown: logs teardown message.

    Model loading happens once at startup — not per request. This keeps
    per-request latency minimal and avoids repeated disk I/O.
    """
    global _model_server

    logger.info("Starting up — loading model artefacts...")
    cfg = _load_deployment_config()

    project_root = Path(__file__).resolve().parents[2]
    model_path = str(project_root / cfg["model"]["model_path"])
    preprocessor_path = str(project_root / cfg["model"]["preprocessor_path"])

    _model_server = ModelServer(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        model_name=cfg["model"]["model_name"],
        model_version=cfg["model"]["model_version"],
        slow_request_threshold_ms=cfg["monitoring"]["slow_request_threshold_ms"],
    )

    try:
        _model_server.load()
        logger.info("Model server ready.")
    except FileNotFoundError as exc:
        logger.error(
            f"Model artefact not found: {exc}. "
            "Run the training pipeline first: "
            "python pipelines/run_training.py --config configs/training_config.yaml"
        )
        # Allow the server to start so /health can report unhealthy state
        # rather than crashing the process entirely.

    yield  # Application runs here

    logger.info("Shutting down model server.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
cfg_for_meta = _load_deployment_config() if Path("configs/deployment_config.yaml").exists() else {}
api_cfg = cfg_for_meta.get("api", {})

app = FastAPI(
    title=api_cfg.get("title", "California Housing Price Predictor"),
    description=api_cfg.get(
        "description", "REST API for predicting median house values in California census blocks."
    ),
    version=api_cfg.get("version", "1.0.0"),
    docs_url=api_cfg.get("docs_url", "/docs"),
    lifespan=lifespan,
)

# CORS middleware — allow all origins in dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time-Ms header to every response."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["Operations"],
)
async def health(response: Response) -> HealthResponse:
    """
    Check whether the service is running and the model is loaded.

    Returns HTTP 200 with ``status: ok`` if healthy.
    Returns HTTP 503 with ``status: degraded`` if the model failed to load.
    """
    loaded = _model_server is not None and _model_server.is_loaded
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=_model_server.model_name if _model_server else "unknown",
        model_version=_model_server.model_version if _model_server else "unknown",
    )
    if not loaded:
        response.status_code = 503
    return body


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict median house value for one census block",
    tags=["Prediction"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error — invalid input features."},
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
)
async def predict(features: HousingFeatures) -> PredictionResponse:
    """
    Predict the median house value for a single California census block group.

    Send a JSON body with the 8 housing features. The response contains the
    predicted value in units of $100,000 and in USD.

    **Example request:**
    ```json
    {
        "MedInc": 3.5,
        "HouseAge": 28.0,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    ```
    """
    if _model_server is None or not _model_server.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    instance = features.model_dump()
    predictions = _model_server.predict([instance])
    pred_val = float(predictions[0])

    if cfg_for_meta.get("monitoring", {}).get("log_predictions", True):
        logger.info(f"Single prediction: {pred_val:.4f} (${pred_val * 100_000:,.0f})")

    return PredictionResponse(
        predicted_value=round(pred_val, 4),
        predicted_value_usd=round(pred_val * 100_000, 2),
        model_name=_model_server.model_name,
        model_version=_model_server.model_version,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict median house values for multiple census blocks",
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Batch exceeds maximum size."},
        422: {"model": ErrorResponse, "description": "Validation error — invalid input features."},
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict median house values for up to 1,000 census block groups in one call.

    Batch prediction is more efficient than calling ``/predict`` in a loop —
    the preprocessing and model forward-pass are vectorised across all samples.

    **Request body:** A JSON object with an ``instances`` list, where each
    item has the same structure as the single-predict endpoint.
    """
    if _model_server is None or not _model_server.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    max_batch = cfg_for_meta.get("api", {}).get("max_batch_size", 1000)
    if len(request.instances) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.instances)} exceeds maximum of {max_batch}.",
        )

    instances = [inst.model_dump() for inst in request.instances]
    predictions = _model_server.predict(instances)

    pred_list = [round(float(p), 4) for p in predictions]
    pred_usd = [round(p * 100_000, 2) for p in pred_list]

    if cfg_for_meta.get("monitoring", {}).get("log_predictions", True):
        import numpy as np

        arr = np.array(pred_list)
        logger.info(
            f"Batch prediction: {len(pred_list)} samples | "
            f"mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}"
        )

    return BatchPredictionResponse(
        predictions=pred_list,
        predictions_usd=pred_usd,
        n_samples=len(pred_list),
        model_name=_model_server.model_name,
        model_version=_model_server.model_version,
    )
