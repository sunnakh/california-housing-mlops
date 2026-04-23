from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def get_xgboost(**kwargs) -> XGBRegressor:

    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",  # Faster than 'auto' for tabular data
        "n_jobs": -1,
    }

    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def get_lightgbm(**kwargs) -> LGBMRegressor:

    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    defaults.update(kwargs)
    return LGBMRegressor(**defaults)


def get_catboost(**kwargs) -> CatBoostRegressor:

    defaults = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
    }
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)
