import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger

logger = get_logger(__name__)

LOG_COLS = ["MedInc", "AveRooms", "Population", "AveOccup"]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio-based interaction features derived from existing columns.

    Creates:
        - rooms_per_household: AveRooms / AveOccup
        - bedrooms_per_room: AveBedrms / AveRooms
        - population_per_household: Population / AveOccup

    Zero denominators are replaced with NaN then filled with column median.
    """
    df = df.copy()

    ave_occup_safe = df["AveOccup"].replace(0, np.nan)
    ave_rooms_safe = df["AveRooms"].replace(0, np.nan)

    df["rooms_per_household"] = df["AveRooms"] / ave_occup_safe
    df["bedrooms_per_room"] = df["AveBedrms"] / ave_rooms_safe
    df["population_per_household"] = df["Population"] / ave_occup_safe

    for col in ["rooms_per_household", "bedrooms_per_room", "population_per_household"]:
        n_nans = int(df[col].isna().sum())
        if n_nans > 0:
            df[col] = df[col].fillna(df[col].median())
            logger.warning(
                f"Filled {n_nans} NaN values in '{col}' with median "
                f"(caused by zero-valued denominator)."
            )

    logger.info(
        "Added interaction features: "
        "rooms_per_household, bedrooms_per_room, population_per_household."
    )
    return df


def add_log_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply log1p transform to skewed numerical columns.

    log1p(x) = log(1 + x) — handles zero values safely.
    Adds new columns with '_log' suffix, keeps originals.

    Args:
        df: Input DataFrame.
        cols: Column names to transform. Must be non-negative.

    Raises:
        ValueError: If any column contains negative values.
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame — skipping.")
            continue
        if (df[col] < 0).any():
            raise ValueError(
                f"Column '{col}' contains negative values. log1p requires non-negative input."
            )
        df[f"{col}_log"] = np.log1p(df[col])
        logger.debug(f"Added log1p transform: '{col}' -> '{col}_log'")

    logger.info(f"Log-transformed {len(cols)} columns: {cols}")
    return df


def build_full_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Steps:
        1. Add interaction features (ratios)
        2. Add log-transformed features for skewed columns

    Args:
        df: Raw DataFrame from load_california_housing().

    Returns:
        Engineered DataFrame with additional feature columns.
    """
    logger.info("Starting feature engineering pipeline...")

    df = add_interaction_features(df)
    df = add_log_features(df, cols=LOG_COLS)

    logger.info(
        f"Feature engineering complete. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns."
    )
    return df
