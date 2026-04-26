
import numpy as np
from sklearn.model_selection import train_test_split

from common.logger import get_logger

logger = get_logger(__name__)


def train_val_test_split(
    x: np.ndarray,
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

    n_total = len(x)
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    val_fraction_of_remaining = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_fraction_of_remaining, random_state=random_state
    )

    n_train, n_val, n_test = len(x_train), len(x_val), len(x_test)
    logger.info(
        f"Data split: train={n_train:,} ({n_train / n_total:.1%}), "
        f"val={n_val:,} ({n_val / n_total:.1%}), "
        f"test={n_test:,} ({n_test / n_total:.1%}). "
        f"Total={n_total:,}."
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
