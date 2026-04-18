from os import path
from pathlib import Path
from typing import cast

import pandas as pd
import numpy as np

import sys

from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch


# Allow imports from the repo root (shared utilities)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COL = "MedHouseVal"


RAW_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "california_housing.csv"
)


def load_california_housing(save_raw: bool = True) -> pd.DataFrame:

    if RAW_DATA_PATH.exists():
        logger.info(f"Loading cached raw data from {RAW_DATA_PATH} ...")

        df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Dataset loaded {df.shape[0]:, } rows * {df.shape[1]} columns.")

        return df

    logger.info("Fetching California Housing Dataset from sklearn...")
    dataset: Bunch = cast(Bunch, fetch_california_housing(as_frame=True))

    df = dataset.frame.copy()
    df.columns = FEATURE_COLS + TARGET_COL

    logger.info(
        f"Dataset loaded: {df.shape[0]:,} rows * {df.shape[1]} columns. "
        f"Target column: {TARGET_COL}"
    )

    # Save raw data so it can be inspected, versioned, and reused
    if save_raw:
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Raw data saved to {RAW_DATA_PATH}")

    return df


def validate_data(df: pd.DataFrame) -> None:

    # Column check
    expected_columns = FEATURE_COLS + TARGET_COL
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Null check
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Dataset contains nulls in columns: {cols_with_nulls.to_dict()}"
        )

    # Target range check
    target_min = df[TARGET_COL].min()
    target_max = df[TARGET_COL].max()

    if target_min < 0.1 or target_max > 6.0:
        raise ValueError(
            f"Target '{TARGET_COL}' out of expected range."
            f"Got min= {target_min:.3f}, max= {target_max:.3f}. "
            "Expected [0.1, 6.0]."
        )

    # Row count check
    if len(df) < 10000:
        raise ValueError(
            f"Dataset too small: {len(df):,} rows, Expected at least 10,000 rows."
        )

    logger.info(
        f"Data Validation passed. Shape= {df.shape}, "
        f"target range= [{target_min:.3f}, {target_max}:.3f]"
        "nulls=0."
    )
