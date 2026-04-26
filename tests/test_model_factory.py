import numpy as np
import pytest

from common.logger import get_logger
from src.models.model_factory import MODEL_REGISTRY, get_model, list_available_models

logger = get_logger(__name__)


@pytest.fixture
def small_regression_data():
    """Tiny training set for smoke-testing model fitting."""
    np.random.seed(42)
    x = np.random.randn(100, 5)
    y = np.random.randn(100)

    return x, y


class TestGetModel:
    """Tests for the get_model() factory units"""

    def test_get_model_returns_estimator(self, small_regression_data):
        """get_model('linear') must return a fitted-ready sklearn estimators"""
        model = get_model("linear")

        assert hasattr(model, "fit"), "Model must have a fit() method."
        assert hasattr(model, "predict"), "Model must have a predict() method."

    def test_unknown_model_raises_valuerror(self):
        """get_model() with an unknown name must raise ValueError"""

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistence_model_xyza")

    def test_get_model_accepts_hyperparameters(self):
        """get_model() must pass hyperparameters to the factory correctly"""
        model = get_model("ridge", config={"alpha": 10.0})
        assert model.alpha == 10.0, f"Expected alpha = 10.0, got {model.alpha}"

    def test_get_model_xgboost_with_config(self):
        """XGBoost model must accept hyperparameter config without error."""

        model = get_model("xgboost", config={"n_estimators": 50, "max_depth": 3})
        assert model.n_estimators == 50

    def test_get_dummy_model(self, small_regression_data):
        """DummyRegressor must be instantiated and fit without error"""

        x, y = small_regression_data
        model = get_model("dummy")
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape == (len(x),)


class TestModelRegistry:
    """Tests verifying the registry structure."""

    def test_all_models_in_registry(self):
        """Every value in MODEL_REGISTRY must be a callable (factory function)"""

        for name, factory in MODEL_REGISTRY.items():
            assert callable(factory), f"Registry entry: '{name}' is not callable."

    def test_list_available_models_returns_sorted_list(self):
        """list_available_models() must return a sorted list of strings."""
        models = list_available_models()
        assert isinstance(models, list)
        assert models == sorted(models), "Model list must be sorted."

    def test_registry_contains_expected_models(self):
        """Key models must be present in the registry"""

        expected = {"dummy", "linear", "ridge", "lasso", "random_forest", "xgboost"}
        registered = set(MODEL_REGISTRY.keys())
        missing = expected - registered
        assert not missing, f"Expected models missing from registry: {missing}"

    def test_all_models_can_fit_and_predict(self, small_regression_data):
        """All models in the registry must fit and predict without raising"""

        x, y = small_regression_data

        quick_params = {
            "random_forest": {"n_estimators": 5},
            "extra_trees": {"n_estimators": 5},
            "xgboost": {"n_estimators": 5},
            "lightgbm": {"n_estimators": 5},
            "catboost": {"iterations": 5},
            "decision_tree": {"max_depth": 3},
        }

        for name in MODEL_REGISTRY:
            params = quick_params.get(name, {})
            model = get_model(name=name, config=params)
            model.fit(x, y)
            preds = model.predict(x)
            assert preds.shape == (len(x),), (
                f"Model '{name}' prediction shape mismatch: {preds.shape}"
            )
