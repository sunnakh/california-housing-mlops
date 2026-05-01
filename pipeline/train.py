"""
Training pipeline for California Housing regression models.

Steps:
    1. Load config
    2. Load and split data
    3. Preprocess (fit on train only)
    4. Train all models — log each run to MLflow
    5. Compare results — pick best model by RMSE
    6. Save best model + preprocessor to disk
    7. Register best model to MLflow Model Registry

Usage:
    python pipeline/train.py
    python pipeline/train.py --config configs/training_config.yaml
"""

import argparse
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

from common.logger import get_logger
from common.reproducibility import set_seed
from src.data.loader import TARGET_COL, load_california_housing
from src.data.preprocessor import RegressionPreprocessor
from src.data.splitter import train_val_test_split
from src.evaluation.comparison import ModelComparison
from src.models.model_factory import MODEL_REGISTRY, get_model
from src.utils.helpers import save_model

logger = get_logger(__name__)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[1] / path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return cfg


def tune_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: int = 300,
) -> dict:
    """Search for best LightGBM hyperparameters using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = get_model("lightgbm", params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(
        f"Optuna tuning complete — best RMSE: {study.best_value:.4f} "
        f"after {len(study.trials)} trials"
    )
    return study.best_params


def run(config_path: str = "configs/training_config.yaml") -> None:

    # ── 1. Config ─────────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    random_state = cfg.get("random_state", 42)
    set_seed(random_state)

    # ── 2. MLflow setup ───────────────────────────────────────────────────────
    tracking_uri = cfg.get("mlflow_tracking_uri", "./experiments/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("experiment_name", "california_housing"))
    logger.info(f"MLflow tracking: {tracking_uri}")

    # ── 3. Load + split data ──────────────────────────────────────────────────
    logger.info("Loading California Housing dataset...")
    df = load_california_housing()
    x = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        x, y, test_size=0.2, val_size=0.1, random_state=random_state
    )
    logger.info(f"Data split — train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

    # ── 4. Preprocess — fit on train only ─────────────────────────────────────
    preprocessor = RegressionPreprocessor()
    x_train_scaled = preprocessor.fit_transform(x_train)
    x_val_scaled = preprocessor.transform(x_val)
    x_test_scaled = preprocessor.transform(x_test)
    logger.info("Preprocessing complete.")

    # ── 5. Tune LightGBM hyperparameters with Optuna ─────────────────────────
    tune_cfg = cfg.get("hyperparameter_tuning", {})
    logger.info("Starting Optuna hyperparameter tuning for LightGBM...")
    best_lgbm_params = tune_lightgbm(
        x_train=x_train_scaled,
        y_train=y_train,
        x_val=x_val_scaled,
        y_val=y_val,
        n_trials=tune_cfg.get("n_trials", 50),
        timeout=tune_cfg.get("timeout_seconds", 300),
    )
    logger.info(f"Best LightGBM params: {best_lgbm_params}")

    # ── 6. Train all models ───────────────────────────────────────────────────
    comparison = ModelComparison()
    models_to_skip = {"dummy"}  # skip baseline for comparison
    model_names = [m for m in MODEL_REGISTRY.keys() if m not in models_to_skip]

    for model_name in model_names:
        logger.info(f"Training: {model_name}")

        with mlflow.start_run(run_name=model_name):
            # Log config params
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("train_size", len(x_train))
            if model_name == "lightgbm":
                mlflow.log_params(best_lgbm_params)

            # Train — use tuned params for lightgbm
            t0 = time.perf_counter()
            params = best_lgbm_params if model_name == "lightgbm" else {}
            model = get_model(model_name, params)
            model.fit(x_train_scaled, y_train)
            train_time = time.perf_counter() - t0

            # Evaluate on val set
            y_pred_val = model.predict(x_val_scaled)
            rmse = float(np.sqrt(np.mean((y_val - y_pred_val) ** 2)))
            mae = float(np.mean(np.abs(y_val - y_pred_val)))
            r2 = float(
                1 - np.sum((y_val - y_pred_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
            )

            metrics = {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_metric("train_time_s", round(train_time, 3))

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Register in comparison
            comparison.add_results(
                model_name=model_name,
                metrics_dict=metrics,
                train_time=train_time,
            )

            logger.info(
                f"{model_name} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
                f"R²: {r2:.4f}, time: {train_time:.2f}s"
            )

    # ── 6. Pick best model ────────────────────────────────────────────────────
    best_name = comparison.get_best_model(metrics="rmse")
    logger.info(f"Best model: {best_name}")

    # ── 7. Retrain best model on train+val, evaluate on test ─────────────────
    x_trainval = np.vstack([x_train_scaled, x_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    best_model = get_model(best_name)
    best_model.fit(x_trainval, y_trainval)

    y_pred_test = best_model.predict(x_test_scaled)
    test_rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
    test_r2 = float(
        1 - np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    )

    logger.info(f"Best model test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # Log final result
    with mlflow.start_run(run_name=f"{best_name}_final"):
        mlflow.log_param("model_name", best_name)
        mlflow.log_param("stage", "final")
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name="california_housing_regressor",
        )

    # ── 8. Save artifacts to disk ─────────────────────────────────────────────
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / cfg["model_selection"]["best_model_path"]
    preprocessor_path = model_path.parent / "preprocessor.joblib"

    save_model(best_model, model_path)
    save_model(preprocessor, preprocessor_path)

    logger.info(f"Best model saved to {model_path}")
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    run(config_path=args.config)
