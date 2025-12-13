"""Feature extraction and preprocessing modules."""

from src.features.preprocessing import extract_features
from src.features.derivatives import (
    compute_velocity,
    compute_acceleration,
    compute_speed,
    compute_distance_from_origin,
)
from src.features.normalization import normalize_zscore, normalize_minmax

__all__ = [
    "extract_features",
    "compute_velocity",
    "compute_acceleration",
    "compute_speed",
    "compute_distance_from_origin",
    "normalize_zscore",
    "normalize_minmax",
]

