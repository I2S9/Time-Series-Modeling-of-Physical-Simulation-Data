"""Normalization utilities for time series data."""

import numpy as np


def normalize_minmax(data: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: Input array to normalize.
        axis: Axis along which to compute min and max (default: 0 for time series).
    
    Returns:
        Tuple of (normalized_data, min_values, max_values) for potential denormalization.
    """
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    normalized = (data - data_min) / data_range
    
    return normalized, data_min.squeeze(), data_max.squeeze()


def normalize_zscore(data: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data using z-score (zero mean, unit variance).
    
    Args:
        data: Input array to normalize.
        axis: Axis along which to compute mean and std (default: 0 for time series).
    
    Returns:
        Tuple of (normalized_data, mean_values, std_values) for potential denormalization.
    """
    data_mean = np.mean(data, axis=axis, keepdims=True)
    data_std = np.std(data, axis=axis, keepdims=True)
    
    data_std[data_std == 0] = 1.0
    
    normalized = (data - data_mean) / data_std
    
    return normalized, data_mean.squeeze(), data_std.squeeze()


def denormalize_minmax(normalized_data: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    """Denormalize min-max normalized data.
    
    Args:
        normalized_data: Normalized data in [0, 1] range.
        data_min: Minimum values used for normalization.
        data_max: Maximum values used for normalization.
    
    Returns:
        Denormalized data in original scale.
    """
    data_min = np.asarray(data_min)
    data_max = np.asarray(data_max)
    
    if data_min.ndim < normalized_data.ndim:
        data_min = np.expand_dims(data_min, axis=0)
    if data_max.ndim < normalized_data.ndim:
        data_max = np.expand_dims(data_max, axis=0)
    
    data_range = data_max - data_min
    return normalized_data * data_range + data_min


def denormalize_zscore(normalized_data: np.ndarray, data_mean: np.ndarray, data_std: np.ndarray) -> np.ndarray:
    """Denormalize z-score normalized data.
    
    Args:
        normalized_data: Z-score normalized data.
        data_mean: Mean values used for normalization.
        data_std: Standard deviation values used for normalization.
    
    Returns:
        Denormalized data in original scale.
    """
    data_mean = np.asarray(data_mean)
    data_std = np.asarray(data_std)
    
    if data_mean.ndim < normalized_data.ndim:
        data_mean = np.expand_dims(data_mean, axis=0)
    if data_std.ndim < normalized_data.ndim:
        data_std = np.expand_dims(data_std, axis=0)
    
    return normalized_data * data_std + data_mean

