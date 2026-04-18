from pathlib import Path
from typing import Any

from common.logger import get_logger

import pandas as pd
import numpy as np

logger = get_logger(__name__)


def load_csv(path: str | Path, **kwargs: Any):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path, **kwargs)

    if df.empty:
        raise ValueError(f"Loaded DataFrame from {path} is empty")

    logger.info(f"Loaded {df.shape[0]:,} rows * {df.shape[1]} columns from {path.name}")
    return df
