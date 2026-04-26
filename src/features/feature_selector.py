from typing import cast

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from common.logger import get_logger

logger = get_logger(__name__)


def select_by_variance(x: np.ndarray, threshold: float = 0.01) -> np.ndarray:

    selector = VarianceThreshold(threshold=threshold)
    x_filtered = selector.fit_transform(x)
    n_removed = x.shape[1] - x_filtered.shape[1]
    logger.info(
        f"Variance threshold ({threshold}): removed {n_removed} features, "
        f"{x_filtered.shape[1]} remaining."
    )
    return x_filtered


def select_by_correlation(df: pd.DataFrame, target_col: str, threshold: float = 0.1) -> list[str]:

    feature_cols = [col for col in df.columns if col != target_col]
    corr_input = cast(pd.DataFrame, df[feature_cols + [target_col]])
    corr_matrix = corr_input.corr()
    target_corr = cast(pd.Series, corr_matrix[target_col])

    selected: list[str] = []
    for name, value in target_corr.items():
        if name == target_col:
            continue
        if pd.notna(value) and abs(float(value)) >= threshold:
            selected.append(str(name))

    dropped = [col for col in feature_cols if col not in selected]

    logger.info(
        f"Correlation selection (threshold={threshold}): "
        f"Kept {len(selected)} / {len(feature_cols)} features. "
        f"Dropped: {dropped}"
    )

    return selected


def select_by_importance(
    x: np.ndarray,
    y: np.ndarray,
    model,
    top_n: int = 10,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    model.fit(x, y)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have feature_importances_. "
            f"Use a tree-based model."
        )

    importances = model.feature_importances_
    top_n = min(top_n, x.shape[1])
    indices = np.argsort(importances)[::-1][:top_n]

    if feature_names is not None:
        if len(feature_names) != x.shape[1]:
            raise ValueError("feature_names length must match X.shape[1].")
        selected_names = [feature_names[i] for i in indices]
        logger.info(f"Importance selection: top {top_n} features: {selected_names}")
    else:
        logger.info(f"Importance selection: top {top_n} feature indices: {indices.tolist()}")

    return x[:, indices]
