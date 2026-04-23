import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger

logger = get_logger(__name__)


class RegressionPreprocessor:
    def __init__(self, copy: bool = True) -> None:
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._n_features_in: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RegressionPreprocessor":
        self._scaler.fit(X=x)
        self._is_fitted = True
        self._n_features_in = x.shape[1]

        assert self._scaler.mean_ is not None
        logger.info(
            f"Preprocessor fitted on {x.shape[0]:,} samples, "
            f"{x.shape[1]} features"
            f"Means: {self._scaler.mean_.round(4).tolist()}"
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "RegressionPreprocessor.transform() called before fit(). Call fit(X_train) first."
            )

        if self._n_features_in is not None and x.shape[1] != self._n_features_in:
            raise ValueError(f"Expected {self._n_features_in} features, got {x.shape[1]}.")

        logger.debug(f"Transforming {x.shape[0]:,} samples.")
        return self._scaler.transform(x)

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:

        return self.fit(x, y).transform(x)

    def get_scale_params(self) -> dict:

        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_scale_params().")

        assert self._scaler.mean_ is not None and self._scaler.scale_ is not None
        return {
            "mean": self._scaler.mean_.tolist(),
            "std": self._scaler.scale_.tolist(),
        }
