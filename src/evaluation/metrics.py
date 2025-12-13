"""Evaluation metrics for time series forecasting."""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error (MSE).
    
    Args:
        y_true: True values with shape (n_samples, n_features) or (n_samples,).
        y_pred: Predicted values with same shape as y_true.
    
    Returns:
        Mean squared error as a scalar.
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error (RMSE).
    
    Args:
        y_true: True values with shape (n_samples, n_features) or (n_samples,).
        y_pred: Predicted values with same shape as y_true.
    
    Returns:
        Root mean squared error as a scalar.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error (MAE).
    
    Args:
        y_true: True values with shape (n_samples, n_features) or (n_samples,).
        y_pred: Predicted values with same shape as y_true.
    
    Returns:
        Mean absolute error as a scalar.
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute multiple evaluation metrics.
    
    Args:
        y_true: True values with shape (n_samples, n_features) or (n_samples,).
        y_pred: Predicted values with same shape as y_true.
    
    Returns:
        Dictionary containing MSE, RMSE, and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }

