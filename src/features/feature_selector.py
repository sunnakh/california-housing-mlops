from pathlib import Path
import sys
from typing import List, Optional

from optuna import importance
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from common.logger import get_logger


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logger = get_logger(__name__)


def select_by_variance(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:

    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    n_removed = X.shape[1] - X_filtered.shape[1]
    logger.info(
        f"Variance threshold ({threshold}): removed {n_removed} features"
        f"{X_filtered.shape[1]} remaining."
    )
    return X_filtered


def select_by_correlation(
    df: pd.DataFrame, target_col: str, threshold: float = 0.1
) -> List[str]:

    feature_cols = [col for col in df.columns if col != target_col]
    correlations = (
        df[feature_cols + [target_col]].corr()[target_col].drop(target_col).abs()
    )
    selected = correlations[correlations >= threshold.index.tolist()]
    dropped = [col for col in feature_cols if col not in selected]

    logger.info(
        f"Correlation selection (threshold={threshold})"
        f"Kept {len(selected) / len(feature_cols)} features."
        f"dropped: {dropped}"
    )

    return selected


def select_by_importance(
    X: np.ndarray,
    y: np.ndarray,
    model,
    top_n: int = 10,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    model.fit(X, y)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have feature_importances_. "
            f"Use a tree-based model."
        )

    importances = model.feature_importances_
    top_n = min(top_n, X.shape[1])
    indices = np.argsort(importances)[::-1][:top_n]

    if feature_names is not None:
        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names length must match X.shape[1].")
        selected_names = [feature_names[i] for i in indices]
        logger.info(f"Importance selection: top {top_n} features: {selected_names}")
    else:
        logger.info(
            f"Importance selection: top {top_n} feature indices: {indices.tolist()}"
        )

    return X[:, indices]
