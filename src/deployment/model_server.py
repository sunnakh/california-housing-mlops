"""
ModelServer: loads the trained model + preprocessor once at startup and
serves predictions with minimal overhead per request.

Design principles:
- Load artefacts once (at server startup), not per request.
- Feature engineering is applied here, not in the route handler.
- The preprocessor is the training-time fitted scaler — never refit.
- Prediction latency target: < 5 ms for a single sample.
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from common.logger import get_logger
from src.features.build_features import build_full_feature_pipeline
from src.utils.helpers import load_model

logger = get_logger(__name__)

BASE_FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


class ModelServer:
    """Wraps a fitted regression model and its preprocessor for serving.

    Loaded once at application startup via ``lifespan`` in app.py.
    Thread-safe for concurrent read-only prediction calls."""

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        model_name: str = "housing_regressor",
        slow_request_threshold_ms: float = 50.0,
        model_version: str = "1.0.0",
    ) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_name = model_name
        self.model_version = model_version
        self.slow_request_threshold_ms = slow_request_threshold_ms

        self._model: Any | None = None
        self._preprocessor: Any | None = None
        self._is_loaded = False

    def load(self):
        logger.info(f"Loading model from {self.model_path}")
        self._model = load_model(self.model_path)

        logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
        self._preprocessor = load_model(self.preprocessor_path)

        self._is_loaded = True
        logger.info(
            f"ModelServer ready - {self.model_name} v{self.model_version} "
            f"Model type: {type(self._model).__name__}"
        )

    @property
    def is_loaded(self) -> bool:
        """True if model and preprocessor are loaded and ready."""
        return self._is_loaded

    def predict(self, instances: list[dict]) -> np.ndarray:

        if not self._is_loaded:
            raise RuntimeError("ModelServer.load() must be called before predict().")
        if not instances:
            raise ValueError("Instances list is empty")

        t0 = time.perf_counter()

        # Build DataFrame with base feature columns — feature engineering expands to 14 cols
        df = pd.DataFrame(instances, columns=BASE_FEATURE_COLUMNS)

        # Apply the same feature engineering pipeline used during training
        df_eng = build_full_feature_pipeline(df=df)

        # dropping target if accidentally present

        if "MedHouseVal" in df_eng.columns:
            df_eng = df_eng.drop(columns=["MedHouseVal"])

        x = df_eng.to_numpy(dtype=np.float64)

        model: Any | None = self._model
        preprocessor: Any | None = self._preprocessor
        if model is None or preprocessor is None:
            raise RuntimeError("Model artifacts are not loaded. Call load() before predict().")

        # scaling with the training-time fitted preprocessor
        x_scaled = preprocessor.transform(x)

        predictions = model.predict(x_scaled)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if elapsed_ms > self.slow_request_threshold_ms:
            logger.warning(
                f"Slow prediction: {elapsed_ms:.1f}ms for {len(instances)} sample(s). "
                f"Threshold: {self.slow_request_threshold_ms}ms"
            )
        else:
            logger.debug(f"Predicted {len(instances)} sample(s) in {elapsed_ms:.2f}ms.")
        return np.asarray(predictions)
