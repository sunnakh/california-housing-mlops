import numpy as np
import pytest

from src.data.preprocessor import RegressionPreprocessor
from src.data.splitter import train_val_test_split


@pytest.fixture()
def synthetic_data():
    """Providing small synthetic feature matrices for preprocessor tests."""

    np.random.seed(42)
    x = np.random.randn(500, 8) * np.array([1, 2, 5, 0.5, 100, 3, 10, 5])
    y = np.random.randn(500)

    return x, y


class TestRegressionPreprocessorShape:
    """Tests for shaping correctness after fit_transform and transform."""

    def test_preprocessor_fit_transform_shape(self, synthetic_data):
        x, y = synthetic_data
        preprocessor = RegressionPreprocessor()
        x_scaled = preprocessor.fit_transform(x)
        assert x_scaled.shape == x.shape, (
            f"Shape mismatch: input {x.shape}, output {x_scaled.shape}"
        )

    def test_transform_shape_matches_input(self, synthetic_data):
        """transform output must have the same shape as input for any split."""
        x, y = synthetic_data
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
            x, y, test_size=0.2, val_size=0.1
        )

        preprocessor = RegressionPreprocessor()
        preprocessor.fit(x_train)
        x_val_scaled = preprocessor.transform(x_val)
        x_test_scaled = preprocessor.transform(x_test)

        assert x_val_scaled.shape == x_val.shape
        assert x_test_scaled.shape == x_test.shape


class TestFitOnTest:
    """Tests that enforce the no-leakage constraint."""

    def test_preprocessor_no_fit_on_test(self, synthetic_data):
        x, y = synthetic_data

        x_train = x[:400]
        x_test = x[400:] + 100

        preprocessor = RegressionPreprocessor()
        preprocessor.fit(x_train)

        # transform x_test to confirm it doesn't affect the fitted scaler params
        preprocessor.transform(x_test)

        scale_params = preprocessor.get_scale_params()
        train_mean = x_train.mean(axis=0)

        for i, (stored, actual) in enumerate(zip(scale_params["mean"], train_mean, strict=True)):
            assert abs(stored - actual) < 0.1, (
                f"Feature {i}: stored mean {stored:.4f} != train mean {actual:.4f}. "
                f"Scaler may have been contaminated by test data."
            )

    def test_transform_before_fit_raises(self, synthetic_data):
        """Calling transform() before fit() must raise RuntimeError."""
        x, _ = synthetic_data
        preprocessor = RegressionPreprocessor()

        with pytest.raises(RuntimeError, match="fit"):
            preprocessor.transform(x)

    def test_scaled_values_have_zero_mean_and_unit_variance(self, synthetic_data):
        """After fit_transform on training data, columns should be ~N(0,1)."""

        x, _ = synthetic_data
        preprocessor = RegressionPreprocessor()
        x_scaled = preprocessor.fit_transform(x)

        col_mean = x_scaled.mean(axis=0)
        col_std = x_scaled.std(axis=0)

        np.testing.assert_allclose(
            col_mean, 0, atol=1e-10, err_msg="Column means should be ~0 after StandardScaler."
        )

        np.testing.assert_allclose(
            col_std, 1, atol=1e-10, err_msg="Column stds should be ~1 after StandardScaler."
        )

    def test_wrong_feature_count_raises(self, synthetic_data):
        """transform() with wrong feature count must raise ValueError."""

        x, _ = synthetic_data
        preprocessor = RegressionPreprocessor()
        preprocessor.fit(x)

        x_wrong = np.random.randn(50, x.shape[1] + 3)
        with pytest.raises(ValueError, match="Expected"):
            preprocessor.transform(x_wrong)
