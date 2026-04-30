import numpy as np

from src.evaluation.metrics import compute_regression_metrics


class TestRegressionMetrics:
    def test_rmse_is_zero_for_perfect_predictions(self):
        """RMSE must be exactly 0.0 when predictions equal ground truth."""

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        assert metrics["rmse"] == 0.0, f"Expected RMSE = 0, got {metrics['rmse']}"
        assert metrics["mae"] == 0.0, f"Expected MAE = 0, got {metrics['mae']}"

    def test_metrics_returns_all_required_keys(self):
        """Return dict must contain rmse, mae, r2, mape."""

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        required_keys = {"rmse", "r2", "mae", "mape"}
        assert required_keys.issubset(set(metrics.keys())), (
            f"Missing keys: {required_keys - set(metrics.keys())}"
        )

    def test_r2_range(self):
        """R2 must be <= 1.0 (can be negative for poor models)."""

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        assert metrics["r2"] <= 1.0, f"R2 must be <= 1.0, got {metrics['r2']}"

    def test_r2_is_one_for_perfect_predictions(self):
        """R2 must be exactly 1.0 for perfect predictions."""

        y_true = np.linspace(1, 10, 50)
        y_pred = y_true.copy()
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        assert abs(metrics["r2"] - 1.0) < 1e-10, f"Expected R2 = 1.0, got {metrics['r2']}"

    def test_rmse_is_always_non_negative(self):
        """RMSE must be >= 0 for any input."""

        y_true = np.random.randn(200)
        y_pred = np.random.rand(200)
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        assert metrics["rmse"] >= 0, f"RMSE was negative: {metrics['rmse']}"

    def test_mae_is_always_non_negative(self):
        """MAE must be >= 0 for any input."""

        y_true = np.array([5.0, 3.0, 1.0])
        y_pred = np.array([3.0, 5.0, 3.0])
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        assert metrics["mae"] >= 0, f"MAE was negative: {metrics['mae']}"

    def test_rmse_greater_or_equal_mae(self):
        """RMSE >= MAE always holds (squared penalty amplifies large errors)."""

        np.random.seed(7)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.5
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
        assert metrics["rmse"] >= metrics["mae"], (
            f"RMSE ({metrics['rmse']}) < MAE ({metrics['mae']}) — impossible."
        )

    def test_metrics_values_are_floats(self):
        """All metric values must be Python floats (not numpy scalars)."""

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

        for key, val in metrics.items():
            if val != val:
                continue
            assert isinstance(val, float), f"Metric '{key}' must be float, got {type(val)}"
