from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.figure
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from common.logger import get_logger

logger = get_logger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (10, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def _save_or_show(fig: matplotlib.figure.Figure, output_path: str | Path | None) -> None:
    """Helper: save figure if output_path given, otherwise show it."""

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), bbox_inches="tight")
        logger.info(f"Figure saved to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    output_path: str | Path | None = None,
) -> None:

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importances (tree impurity)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_.flatten())
        title = "Feature Importances (|coefficient|)"
    else:
        logger.warning("Model has neither feature_importances_ nor coef_. Cannot plot.")
        return

    indices = np.argsort(importances)[-top_n:][::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(range(len(sorted_names)), sorted_vals[::-1], color="steelblue")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_learning_curve(
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    output_path: str | Path | None = None,
) -> None:
    """Plot train and validation scores as a function of training set size.

    Useful for diagnosing high bias (both scores low) vs high variance
    (large gap between train and val scores).
    """

    train_sizes, train_scores, val_scores, *_ = learning_curve(
        estimator=estimator,
        X=x,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = np.mean(-train_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    val_mean = np.mean(-val_scores, axis=1)
    val_std = np.std(-val_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, "o-", color="royalblue", label="Training score")
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue"
    )
    ax.plot(train_sizes, val_mean, "o-", color="darkorange", label="Cross-val score")
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="darkorange"
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring.replace("neg_", "").replace("_", " ").title())
    ax.set_title("Learning Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    output_path: str | Path | None = None,
) -> None:

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    _save_or_show(fig=fig, output_path=output_path)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path | None = None,
) -> None:

    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=12, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    # Residual distribution
    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    _save_or_show(fig, output_path)


def actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path | None = None
) -> None:
    """Scatter plot of actual vs predicted values with a perfect-fit reference line."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color="steelblue")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Actual vs Predicted")
    fig.tight_layout()
    _save_or_show(fig, output_path=output_path)
