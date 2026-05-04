import time
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.model_factory import get_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


SUPPORTED_MODELS = {"xgboost", "lightgbm", "random_forest"}


def suggested_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """returning model specific hyperparameters sampled by Optuna"""

    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, *range(3, 21)],
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    raise ValueError(f"Unsupported model: {model_name}. Supported: {SUPPORTED_MODELS}")


def objective(
    trial: optuna.Trial,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """training one trial model and returning validation RMSE"""

    params = suggested_params(trial=trial, model_name=model_name)

    with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_params(params=params)

        model = get_model(name=model_name, config=params)

        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - start

        y_pred_val = model.predict(x_val)

        val_rmse = np.sqrt(mean_squared_error(y_true=y_val, y_pred=y_pred_val))
        val_mae = mean_absolute_error(y_true=y_val, y_pred=y_pred_val)
        val_r2 = r2_score(y_true=y_val, y_pred=y_pred_val)

        mlflow.log_metrics(
            {
                "val_rmse": round(val_rmse, 6),
                "val_mae": round(val_mae, 6),
                "val_r2": round(val_r2, 6),
                "train_time": round(train_time, 3),
            }
        )

    return val_rmse


def tune_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    timeout: int | None = None,
    direction: str = "minimize",
) -> tuple[dict[str, Any], float, optuna.Study]:
    """running one Optuna study for a model and returning best params and score."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model name: {model_name}. Supported model names: {SUPPORTED_MODELS}"
        )

    study = optuna.create_study(study_name=f"{model_name}_tuning", direction=direction)

    with mlflow.start_run(run_name=f"{model_name}_optuna_tuning", nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("timeout", timeout)
        mlflow.log_param("direction", direction)
        mlflow.log_param("metric", "val_rmse")

        study.optimize(
            lambda trial: objective(
                trial=trial,
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            ),
            n_trials=n_trials,
            timeout=timeout,
        )

        mlflow.log_metric("best_val_rmse", round(float(study.best_value), 6))
        mlflow.log_params({f"best_{name}": value for name, value in study.best_params.items()})

    return study.best_params, float(study.best_value), study
