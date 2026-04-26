from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from common.logger import get_logger

logger = get_logger(__name__)


class ModelComparison:
    def __init__(self):
        self._results: dict[str, dict[str, Any]] = {}
        self._best_params: dict[str, dict[str, Any]] = {}

    def store_best_params(self, model_name: str, params: dict[str, Any]) -> None:
        self._best_params: dict[str, dict[str, Any]]
        self._best_params[model_name] = params

    def add_results(
        self,
        model_name: str,
        metrics_dict: dict[str, Any],
        train_time: float,
    ):

        if model_name in self._results:
            logger.warning(
                f"Model '{model_name}' is already registered, Overwriting previous result."
            )

        self._results[model_name] = {**metrics_dict, "train_time_s": round(train_time, 3)}
        logger.info(
            f"Registered result for '{model_name}': "
            f"RMSE={metrics_dict.get('rmse', 'N/A')}, "
            f"R²={metrics_dict.get('r2', 'N/A')}, "
            f"train={train_time:.2f}s"
        )

    def get_best_model(self, metrics: str = "rmse") -> str:

        if not self._results:
            raise ValueError("No results registered. Call add_result() first.")

        higher_is_better = {"r2"}
        values = {name: res[metrics] for name, res in self._results.items() if metrics in res}

        if not values:
            raise ValueError(
                f"Metric '{metrics}' not found in any registered results. "
                f"Available: {set(list(self._results.values())[0].keys())}"
            )

        if metrics in higher_is_better:
            best_name = max(values, key=lambda k: values[k])
        else:
            best_name = min(values, key=lambda k: values[k])

        logger.info(f"Best model by '{metrics}': '{best_name}' ({metrics}={values[best_name]:.4f})")

        return best_name

    def to_dataframe(self) -> pd.DataFrame:

        df = pd.DataFrame(self._results).T
        df.index.name = "Model"

        if "rmse" in df.columns:
            df = df.sort_values("rmse", ascending=True)

        return df

    def plot_comparison(
        self,
        metric: str = "rmse",
        save_path: str | None = None,
    ) -> None:
        """
        Plot a horizontal bar chart comparing all models on a single metric.

        Args:
            metric: Metric to visualize. Default 'rmse'.
            save_path: If provided, save figure to this path.
        """
        df = self.to_dataframe()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not in results. Available: {list(df.columns)}")

        values = df[metric].astype(float)
        best_model = self.get_best_model(metric)

        colors = ["steelblue" if name != best_model else "darkorange" for name in values.index]

        fig, ax = plt.subplots(figsize=(10, max(3, len(values) * 0.5)))
        bars = ax.barh(values.index[::-1], values.values[::-1], color=colors[::-1])
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Model Comparison — {metric.upper()}\n(orange = best)")

        # Add value labels on bars
        for bar, val in zip(bars, values.values[::-1], strict=True):
            ax.text(
                bar.get_width() + max(values) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        fig.tight_layout()

        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")

        plt.show()
        plt.close(fig)
