import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logger import get_logger
from src.models.baseline import get_dummy_regressor, get_linear_regression
from src.models.gradient_boosting import get_catboost, get_lightgbm, get_xgboost
from src.models.regularized import get_elastic_et, get_lasso, get_ridge
from src.models.tree_based import get_decision_tree, get_extra_tree, get_random_forest

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, Any] = {
    "dummy": get_dummy_regressor,
    "linear": get_linear_regression,
    "ridge": get_ridge,
    "lasso": get_lasso,
    "elasticnet": get_elastic_et,
    "decision_tree": get_decision_tree,
    "random_forest": get_random_forest,
    "extra_trees": get_extra_tree,
    "xgboost": get_xgboost,
    "lightgbm": get_lightgbm,
    "catboost": get_catboost,
}


def get_model(name: str, config: dict[str, Any] | None = None):

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")

    factory = MODEL_REGISTRY[name]
    params = config or {}

    # Baseline models (dummy, linear) take no hyperparameters
    if name in ("dummy", "linear"):
        model = factory()
    else:
        model = factory(**params)

    logger.info(f"Created model '{name}' with params: '{params}' .")

    return model


def list_available_models() -> list[str]:

    return sorted(MODEL_REGISTRY.keys())
