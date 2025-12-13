"""Main preprocessing pipeline for trajectory data."""

import numpy as np

from src.features.derivatives import (
    compute_velocity,
    compute_acceleration,
    compute_speed,
    compute_distance_from_origin,
)
from src.features.normalization import normalize_zscore, normalize_minmax


def extract_features(
    positions: np.ndarray,
    include_velocity: bool = True,
    include_acceleration: bool = False,
    include_speed: bool = True,
    include_distance: bool = True,
    normalize: bool = True,
    normalization_method: str = "zscore",
) -> tuple[np.ndarray, dict]:
    """Extract features from position trajectories.
    
    Args:
        positions: Array of positions with shape (n_steps, dimension) or (n_steps, n_particles, dimension).
        include_velocity: Whether to include velocity features.
        include_acceleration: Whether to include acceleration features.
        include_speed: Whether to include speed features.
        include_distance: Whether to include distance from origin.
        normalize: Whether to normalize the features.
        normalization_method: Normalization method ("zscore" or "minmax").
    
    Returns:
        Tuple of (feature_array, metadata_dict) where metadata contains normalization parameters.
    """
    features_list = []
    metadata = {}
    
    if include_velocity:
        velocity = compute_velocity(positions)
        features_list.append(velocity)
        metadata["has_velocity"] = True
    else:
        metadata["has_velocity"] = False
    
    if include_acceleration:
        acceleration = compute_acceleration(positions)
        features_list.append(acceleration)
        metadata["has_acceleration"] = True
    else:
        metadata["has_acceleration"] = False
    
    if include_speed:
        speed = compute_speed(positions)
        if speed.ndim == 1:
            speed = speed.reshape(-1, 1)
        features_list.append(speed)
        metadata["has_speed"] = True
    else:
        metadata["has_speed"] = False
    
    if include_distance:
        distance = compute_distance_from_origin(positions)
        if distance.ndim == 1:
            distance = distance.reshape(-1, 1)
        features_list.append(distance)
        metadata["has_distance"] = True
    else:
        metadata["has_distance"] = False
    
    if len(features_list) == 0:
        raise ValueError("At least one feature type must be included")
    
    min_length = min(f.shape[0] for f in features_list)
    features_list = [f[:min_length] for f in features_list]
    
    features = np.concatenate(features_list, axis=-1)
    
    if normalize:
        if normalization_method == "zscore":
            features, mean, std = normalize_zscore(features, axis=0)
            metadata["normalization"] = "zscore"
            metadata["normalization_params"] = {"mean": mean, "std": std}
        elif normalization_method == "minmax":
            features, data_min, data_max = normalize_minmax(features, axis=0)
            metadata["normalization"] = "minmax"
            metadata["normalization_params"] = {"min": data_min, "max": data_max}
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
    else:
        metadata["normalization"] = None
    
    metadata["feature_shape"] = features.shape
    metadata["n_features"] = features.shape[-1]
    
    return features, metadata

