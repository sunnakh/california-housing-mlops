import sys
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.eval_utils import regression_metrics
from common.logger import get_logger

logger = get_logger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:

    return regression_metrics(y_true=y_true, y_pred=y_pred)


def log_metrics_to_mlflow(
    metrics_dict: dict[str, float], step: int | None = None, prefix: str = ""
) -> None:
    for key, value in metrics_dict.items():
        metrics_name = f"{prefix}{key}" if prefix else key
        mlflow.log_metric(metrics_name, value=value, step=step)
        logger.debug(f"MLFlow logged: {metrics_name}={value}")

    logger.info(
        f"Logged {len(metrics_dict)} metrics to MLflow"
        + (f" with prefix '{prefix}'" if prefix else "")
        + (f" at step {step}" if step is not None else "")
    )
