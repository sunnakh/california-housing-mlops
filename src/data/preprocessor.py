from typing import Optional

import pandas as pd
import numpy as np

from pathlib import Path


import sys

from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger

logger = get_logger(__name__)


class RegressionPreprocessor:

    def __init__(self, copy: bool = True) -> None:
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._n_features_in: Optional[int] = None

    def fit(
        self, X=np.ndarray, y: Optional[np.ndarray] = None
    ) -> "RegressionPreprocessor":
        self._scaler.fit(X=X)
        self._is_fitted = True
        self._n_features_in = X.shape[1]

        assert self._scaler.mean_ is not None
        logger.info(
            f"Preprocessor fitted on {X.shape[0]:,} samples, "
            f"{X.shape[1]} features"
            f"Means: {self._scaler.mean_.round(4).tolist()}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "RegressionPreprocessor.transform() called before fit(). "
                "Call fit(X_train) first."
            )

        if self._n_features_in is not None and X.shape[1] != self._n_features_in:
            raise ValueError(
                f"Expected {self._n_features_in} features, got {X.shape[1]}."
            )

        logger.debug(f"Transforming {X.shape[0]:,} samples.")
        return self._scaler.transform(X)

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        return self.fit(X, y).transform(X)

    def get_scale_params(self) -> dict:

        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_scale_params().")

        assert self._scaler.mean_ is not None and self._scaler is not None
        return {
            "mean": self._scaler.mean_.tolist(),
            "std": self._scaler.scale_.tolist(),
        }
