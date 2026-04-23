import os
import random

import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for all libraries used across the ML Mastery System."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Global seed set to {seed} (random, numpy, PYTHONHASHSEED)")
