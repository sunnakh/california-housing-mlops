"""Optuna tuning pipeline.

Tunes supported model families, logs trial results to MLflow, and writes each
model's best hyperparameters back into the training config.

Run as a module from the project root:

- `python -m pipeline.tune --config configs/tuning_config.yaml \
    --training-config configs/training_config.yaml`
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mlflow
import ruamel.yaml

from common.logger import get_logger
from pipeline.train import load_config, prepare_data
from src.tuning.optuna_tuner import SUPPORTED_MODELS, tune_model

logger = get_logger(__name__)


def resolve_path(path: str) -> Path:
    """Resolve paths from cwd first, then from project root."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[1] / path


def normalize_timeout(timeout: Any) -> int | None:
    """Convert YAML timeout values into Optuna-compatible timeout values.

    Args:
        timeout: Raw timeout value from config.

    Returns:
        Integer timeout in seconds, or None for no timeout.
    """
    if timeout is None:
        return None

    if isinstance(timeout, str) and timeout.lower() in {"", "none", "null"}:
        return None

    return int(timeout)


def run_tuning(
    config_path: str,
    training_config_path: str,
) -> dict[str, dict[str, Any]]:
    """Run one Optuna study per configured model.

    Args:
        config_path: Path to tuning config YAML.
        training_config_path: Path to training config YAML used for data prep
            and MLflow settings.

    Returns:
        Mapping from model name to best hyperparameters.
    """

    tuning_cfg = load_config(config_path)
    training_cfg = load_config(training_config_path)

    tracking_uri = training_cfg.get("mlflow_tracking_uri", "./experiments/mlruns")
    experiment_name = training_cfg.get("experiment_name", "california_housing")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    (
        x_train_scaled,
        x_val_scaled,
        _x_test_scaled,
        y_train,
        y_val,
        _y_test,
        _preprocessor,
    ) = prepare_data(training_cfg)

    n_trials = int(tuning_cfg.get("n_trials", 30))
    timeout = normalize_timeout(timeout=tuning_cfg.get("timeout", 300))
    direction = str(tuning_cfg.get("direction", "minimize"))
    metric = str(tuning_cfg.get("metric", "val_rmse"))
    models_to_tune = list(tuning_cfg.get("models_to_tune", sorted(SUPPORTED_MODELS)))

    unsupported_models = sorted(set(models_to_tune) - SUPPORTED_MODELS)
    if unsupported_models:
        raise ValueError(
            f"Unsupported models in tuning config: {unsupported_models}. "
            f"Supported models: {sorted(SUPPORTED_MODELS)}"
        )

    best_params_by_model: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="optuna_tuning_summary"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("timeout", timeout)
        mlflow.log_param("direction", direction)
        mlflow.log_param("metric", metric)
        mlflow.log_param("models_to_tune", ",".join(models_to_tune))

        for model_name in models_to_tune:
            logger.info("Starting Optuna tuning for %s", model_name)

            best_params, best_score, _study = tune_model(
                model_name=model_name,
                x_train=x_train_scaled,
                y_train=y_train,
                x_val=x_val_scaled,
                y_val=y_val,
                n_trials=n_trials,
                timeout=timeout,
                direction=direction,
            )

            best_params_by_model[model_name] = best_params
            summary_rows.append(
                {
                    "model_name": model_name,
                    "best_val_rmse": best_score,
                    "best_params": json.dumps(best_params, sort_keys=True),
                }
            )

            mlflow.log_metric(f"{model_name}_best_val_rmse", best_score)
            mlflow.log_params({f"{model_name}_{key}": value for key, value in best_params.items()})

        summary_path = Path("experiments/optuna_tuning_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_name", "best_val_rmse", "best_params"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        mlflow.log_artifact(str(summary_path))

    return best_params_by_model


def write_best_params_to_config(
    best_params: dict[str, dict[str, Any]],
    training_config_path: str,
) -> None:
    """Write tuned params into training_config.yaml while preserving comments.

    Args:
        best_params: Mapping from model name to best hyperparameters.
        training_config_path: Path to training config YAML.
    """
    config_path = resolve_path(training_config_path)
    yaml_parser = ruamel.yaml.YAML()
    yaml_parser.preserve_quotes = True

    with open(config_path) as f:
        cfg = yaml_parser.load(f)

    if cfg.get("models") is None:
        cfg["models"] = {}

    for model_name, params in best_params.items():
        cfg["models"][model_name] = params

    with open(config_path, "w") as f:
        yaml_parser.dump(cfg, f)

    logger.info("Best tuned params written to %s", config_path)


def run(
    config_path: str = "configs/tuning_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
) -> None:
    """Run tuning and write best params to training config.

    Args:
        config_path: Path to tuning config YAML.
        training_config_path: Path to training config YAML.
    """
    best_params = run_tuning(
        config_path=config_path,
        training_config_path=training_config_path,
    )
    write_best_params_to_config(
        best_params=best_params,
        training_config_path=training_config_path,
    )
    logger.info("Optuna tuning pipeline completed successfully.")


if __name__ == "__main__":
    if __package__ in (None, ""):
        raise SystemExit(
            "Run from the project root as a module: "
            "python -m pipeline.tune --config configs/tuning_config.yaml "
            "--training-config configs/training_config.yaml"
        )

    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning.")
    parser.add_argument(
        "--config",
        default="configs/tuning_config.yaml",
        help="Path to tuning config YAML.",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training_config.yaml",
        help="Path to training config YAML that receives tuned params.",
    )
    args = parser.parse_args()

    run(config_path=args.config, training_config_path=args.training_config)
