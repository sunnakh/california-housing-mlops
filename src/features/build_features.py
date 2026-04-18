from pathlib import Path
import sys
from typing import List

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger

logger = get_logger(__name__)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    ave_occup_safe = df["AveOccup"].replace(0, np.nan)
    ave_rooms_safe = df["AveRooms"].replace(0, np.nan)

    df["rooms_per_household"] = df["AveRooms"] / ave_occup_safe
    df["bedrooms_per_room"] = df["AveBedrms"] / ave_rooms_safe
    df["population_per_household"] = df["Population"] / ave_occup_safe

    for col in ["rooms_per_household", "bedrooms_per_room", "population_per_household"]:
        n_nans = df[col].isna().sum()

        if n_nans > 0:
            df[col].fillna(df[col].median(), inplace=True)
            logger.warning(
                f"Filled {n_nans} NaN values in '{col}' with median "
                f"(caused by zero-valued denominator)."
            )

        logger.info(
            "Added interaction features: rooms_per_household, bedrooms_per_room, population_per_household."
        )
        return df

    def add_log_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:

        df = df.copy()
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column not found in DataFrame - skipping")
                continue
            if (df[col] < 0).any():
                raise ValueError(
                    f"Column '{col}' contains negative values. "
                    f"log1p requires non-negative input."
                )
            df[f"{col}_log"] = np.log1p(df[col])
            logger.debug(f"Added log1p transform for '{col}' -> '{col}_log'")

            logger.info(f"Log-transformed {len(cols)} columns: {cols}")
            return df

        def build_full_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:

            logger.info("Starting feature engineering pipeline...")
            df = add_interaction_features(df)
            df = add_log_features(df)
            logger.info(
                f"Feature engineering complete. "
                f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns."
            )
            return df
