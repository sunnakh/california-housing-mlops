from typing import Any, Dict, Literal

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from common.logger import get_logger

logger = get_logger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

    rmse = float(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)))
    mae = float(mean_absolute_error(y_true=y_true, y_pred=y_pred))
    r2 = float(r2_score(y_true=y_true, y_pred=y_pred))
    # MAPE: guard against division by zero
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        mape = float("nan")
        logger.warning("MAPE undefined — all y_true values are zero.")
    else:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
        )
    metrics = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2_score": round(r2, 6),
        "mape": round(mape, 6),
    }

    logger.info(
        f"Regression metrics: RMSE={rmse:.4f}, MAE={mae:.3f}, R2_score={r2:.3f}, MAPE={mape:.2f}"
    )

    return metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "binary",
) -> Dict[str]:

    accuracy = float(accuracy_score(y_true=y_true, y_pred=y_pred))
    precision = float(precision_score(y_true=y_true, y_pred=y_pred))
    recall = float(recall_score(y_true=y_true, y_pred=y_pred))
    f1 = float(f1_score(y_true=y_true, y_pred=y_pred))

    metrics = {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }

    if y_prob is not None:
        try:
            if average == "binary":
                auc = float(roc_auc_score(y_true, y_prob))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            metrics["auc"] = round(auc, 6)
        except ValueError as e:
            logger.warning(f"Could not compute auc: {e}")

        logger.info(
            f"Classification Metrics - Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1_score={f1:.3f}"
        )

    return metrics


def build_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:

    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"

    # Sort by first numeric column (assume lower is better — e.g. RMSE)

    first_col = df.columns[0]
    df = df.sort_values(by=first_col, ascending=True)

    logger.info(f"Comparison table built for {len(df)} models.")

    return df
