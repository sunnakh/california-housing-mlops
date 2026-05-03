import time
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from common.logger import get_logger
from src.data.loader import TARGET_COL, load_california_housing
from src.data.preprocessor import RegressionPreprocessor
from src.data.splitter import train_val_test_split
from src.evaluation.comparison import ModelComparison
from src.features.build_features import build_full_feature_pipeline
from src.models.model_factory import MODEL_REGISTRY, get_model
from src.utils.helpers import save_model

logger = get_logger(__name__)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    config_path = Path(path)

    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[1] / path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Config file loaded from {config_path}")

    return cfg


def prepare_data(
    cfg: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    RegressionPreprocessor,
]:
    """Load, split, and preprocess California Housing data.

    Args:
        cfg: Config dictionary from load_config.

    Returns:
        Tuple of (x_train_scaled, x_val_scaled, x_test_scaled,
                  y_train, y_val, y_test, preprocessor).
    """

    # Loading data from raw data
    df = load_california_housing()
    # Apply feature engineering pipeline (adds derived features)
    df = build_full_feature_pipeline(df=df)
    logger.info(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

    # splitting data into X, Y using target_col
    x = df.drop(columns=[TARGET_COL]).to_numpy()
    y = df[TARGET_COL].to_numpy()

    # split into train val test

    test_size = float(cfg.get("test_size", 0.2))
    val_size = float(cfg.get("val_size", 0.1))
    random_state = int(cfg.get("random_state", 42))
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        x=x, y=y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    logger.info(
        "Split sizes: train=%d val=%d test=%d (test_size=%.2f val_size=%.2f rs=%d)",
        len(x_train),
        len(x_val),
        len(x_test),
        test_size,
        val_size,
        random_state,
    )

    preprocessor = RegressionPreprocessor()
    x_train_scaled = preprocessor.fit_transform(x_train)
    x_val_scaled = preprocessor.transform(x_val)
    x_test_scaled = preprocessor.transform(x_test)
    logger.info(
        "Scaled shapes: train=%s val=%s test=%s",
        x_train_scaled.shape,
        x_val_scaled.shape,
        x_test_scaled.shape,
    )

    return (
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train,
        y_val,
        y_test,
        preprocessor,
    )


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> tuple[str, ModelComparison]:
    """Training all models, log to MLflow, return best model name.

    Args:
        x_train: Scaled training features.
        y_train: Training targets.
        x_val: Scaled validation features.
        y_val: Validation targets.
        cfg: Config dictionary from load_config.

    Returns:
        Tuple of (best_model_name, comparison_object).

    Raises:
        ValueError: If MODEL_REGISTRY is empty.
    """

    comparison = ModelComparison()
    model_names = list(MODEL_REGISTRY.keys())

    if not model_names:
        raise ValueError("Model names not found in Model Registry")

    random_state = int(cfg.get("random_state", 42))
    logger.info("Starting model training for %d models", len(model_names))

    for model_name in model_names:
        with mlflow.start_run(run_name=model_name):
            model_params = cfg.get("models", {}).get(model_name, {})

            model = get_model(name=model_name, config=model_params)

            # resolved params from the actual estimator not just overriding
            resolved_params = model.get_params() if hasattr(model, "get_params") else model_params

            # logging params
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_params(resolved_params)

            # training and timing it
            logger.info(f"Training {model_name}...")
            start = time.perf_counter()
            model.fit(x_train, y_train)
            train_time = time.perf_counter() - start

            # prediction on validation test
            y_pred_val = model.predict(x_val)

            # computing validation metrics
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            mae = mean_absolute_error(y_val, y_pred=y_pred_val)
            r2 = r2_score(y_val, y_pred_val)

            metrics = {"rmse": rmse, "mae": mae, "r2": r2}

            # logging metrics
            mlflow.log_metrics(metrics=metrics)

            # logging model artifacts
            mlflow_sklearn.log_model(model, artifact_path="model")

            # registering and adding the result
            comparison.add_results(
                model_name=model_name, metrics_dict=metrics, train_time=train_time
            )

            comparison.store_best_params(model_name=model_name, params=resolved_params)

            logger.info(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}, time: {train_time:.2f}s")

    best_name = comparison.get_best_model("rmse")
    logger.info("Training complete. Best model by rmse: %s", best_name)

    return best_name, comparison


def save_and_register(
    best_model, preprocessor: RegressionPreprocessor, best_name: str, cfg: dict
) -> None:
    """Save best model and preprocessor to disk, register to MLflow.

    Args:
        best_model: Trained best model object.
        preprocessor: Fitted preprocessor object.
        best_name: Name of the best model.
        cfg: Config dictionary from load_config.
    """

    # Get model file path from config (expects a file path)
    model_file_path = Path(cfg["model_selection"]["best_model_path"])
    logger.info("Saving best model to %s", model_file_path)

    # Save preprocessor next to the model file
    model_dir = model_file_path.parent
    preprocessor_path = model_dir / "preprocessor.joblib"
    logger.info("Saving preprocessor to %s", preprocessor_path)

    # Persist to disk
    save_model(best_model, model_file_path)
    save_model(preprocessor, preprocessor_path)

    # Log final run and register model in MLflow
    with mlflow.start_run(run_name=f"{best_name}_final"):
        mlflow.log_param("model_name", best_name)
        mlflow.log_param("stage", "final")
        mlflow.log_param("model_disk_path", str(model_file_path))
        mlflow.log_param("preprocessor_disk_path", str(preprocessor_path))

        mlflow_sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name="housing_regressor_mlops",
        )


def run(config_path: str = "configs/training_config.yaml") -> None:
    """
    Orchestrate the full training pipeline:
    config -> mlflow setup -> data prep -> model comparison
    -> retrain winner -> save/register artifacts.
    """
    cfg = load_config(path=config_path)

    tracking_uri = cfg.get("mlflow_tracking_uri", "./experiments/mlruns")
    experiment_name = cfg.get("experiment_name", "california_housing")

    # setting mlflow tracking_uri and experiment_name from config
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)
    logger.info(f"MLFlow tracking URI set to: {tracking_uri}")
    logger.info(f"MLFlow experiment sent to: {experiment_name}")

    # preparing data
    (x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test, preprocessor) = (
        prepare_data(cfg=cfg)
    )

    best_name, comparison = train_and_evaluate(
        x_train=x_train_scaled, x_val=x_val_scaled, y_train=y_train, y_val=y_val, cfg=cfg
    )
    logger.info(f"Best model selected {best_name}")

    # retraining best model on train + val test combined
    best_params = comparison._best_params.get(best_name, {})

    x_train_val = np.concatenate([x_train_scaled, x_val_scaled])
    y_train_val = np.concatenate([y_train, y_val])

    best_model = get_model(name=best_name, config=best_params)
    best_model.fit(x_train_val, y_train_val)

    logger.info(f"Retrained best model: {best_model} on combined both train_data + validation_data")

    # save and registering final artifacts

    save_and_register(
        best_model=best_model, preprocessor=preprocessor, best_name=best_name, cfg=cfg
    )

    # final completion log
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    run(config_path=args.config)
