from pathlib import Path

import pandas as pd
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test size: {test_size} + validation size: {val_size} must be < 1.0"
            f"Got {(test_size + val_size):.2f}."
        )

    n_total = len(X)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_fraction_of_remaining = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction_of_remaining, random_state=random_state
    )

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    logger.info(
        f"Data split: train={n_train:,} ({n_train/n_total:.1%}), "
        f"val={n_val:,} ({n_val/n_total:.1%}), "
        f"test={n_test:,} ({n_test/n_total:.1%}). "
        f"Total={n_total:,}."
    )

    return X_train, X_val, X_test, y_train, y_val, y_test



