import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.diagnostic as smd
from statsmodels.api import add_constant

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logger import get_logger
from common.plot_utils import _save_or_show

logger = get_logger(__name__)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None):

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Analysis", fontsize=13, fontweight="bold")

    # Left: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10, color="steelblue", label="Residuals")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2, label="Zero line")
    axes[0].set_xlabel("Predicted values")
    axes[0].set_ylabel("Residuals (actual − predicted)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].legend()

    # Right: Residual histogram
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].axvline(
        residuals.mean(),
        color="orange",
        linestyle="-",
        linewidth=1.2,
        label=f"Mean={residuals.mean():.3f}",
    )
    axes[1].set_xlabel("Residual value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].legend()

    fig.tight_layout()
    _save_or_show(fig, save_path)

    logger.info(
        f"Residual stats — mean={residuals.mean():.4f}, "
        f"std={residuals.std():.4f}, "
        f"max_abs={np.abs(residuals).max():.4f}"
    )


def plot_prediction_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None
) -> None:

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")

    all_vals = np.concatenate([y_true, y_pred])
    min_val, max_val = all_vals.min(), all_vals.max()
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction"
    )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def check_heterosedasticity(y_true: np.ndarray, y_pred: np.ndarray) -> None:

    residuals = y_true - y_pred
    squared_residuals = residuals**2
    # Regress squared residuals on predictions

    x_bp = add_constant(y_pred.reshape(-1, 1))
    bp_test = smd.het_breuschpagan(squared_residuals, x_bp)

    lm_stat = bp_test[0]
    p_value = bp_test[1]

    print("\nBreusch-Pagan Heteroscedasticity Test:")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value:      {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Evidence of heteroscedasticity (p < 0.05).")
        print("Implication: For linear models, standard errors are invalid.")
        print("Consider: WLS, log-transform of target, or a tree model")
    else:
        print("Result: No significant evidence of heteroscedasticity (p >= 0.05)")

    logger.info(
        f"Breusch-Pagan test: LM={lm_stat:.4f}, p={p_value:.6f}, "
        f"heteroscedastic={'yes' if p_value < 0.05 else 'no'}"
    )
