import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib

from common.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(label: str = "Block") -> Generator[None, None, None]:

    start = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{label} completed in {elapsed:.4f}")


def save_model(model: Any, path: str | Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    model_type = type(model).__name__
    file_size_kb = path.stat().st_size / 1024
    logger.info(f"Saved {model_type} to {path} ({file_size_kb:.1f} KB)")


def load_model(path: str | Path) -> Any:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    logger.info(f"Loaded {type(model).__name__} from {path}")
    return model
