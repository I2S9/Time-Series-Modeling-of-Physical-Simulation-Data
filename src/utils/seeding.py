"""Utilities for random seed management to ensure reproducibility."""

import numpy as np
import random


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

