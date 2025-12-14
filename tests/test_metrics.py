"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    compute_metrics,
)


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_mse_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        
        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0
    
    def test_mse_constant_error(self):
        """Test MSE with constant error."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])
        
        mse = mean_squared_error(y_true, y_pred)
        assert mse == 1.0
    
    def test_rmse_relationship_to_mse(self):
        """Test that RMSE is square root of MSE."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        
        assert abs(rmse - np.sqrt(mse)) < 1e-10
    
    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 0.0
    
    def test_mae_constant_error(self):
        """Test MAE with constant error."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])
        
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 1.0
    
    def test_metrics_multidimensional(self):
        """Test metrics with multidimensional arrays."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        assert mse > 0
        assert rmse > 0
        assert mae > 0
        assert rmse == np.sqrt(mse)
    
    def test_compute_metrics_returns_dict(self):
        """Test that compute_metrics returns all metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert isinstance(metrics["mse"], (float, np.floating))
        assert isinstance(metrics["rmse"], (float, np.floating))
        assert isinstance(metrics["mae"], (float, np.floating))
    
    def test_compute_metrics_consistency(self):
        """Test that compute_metrics values match individual functions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        
        metrics = compute_metrics(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        assert abs(metrics["mse"] - mse) < 1e-10
        assert abs(metrics["rmse"] - rmse) < 1e-10
        assert abs(metrics["mae"] - mae) < 1e-10
    
    def test_metrics_symmetric(self):
        """Test that metrics are symmetric (order doesn't matter for error magnitude)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 1.0, 4.0])
        
        mse1 = mean_squared_error(y_true, y_pred)
        mse2 = mean_squared_error(y_pred, y_true)
        
        assert mse1 == mse2
    
    def test_metrics_handles_zeros(self):
        """Test that metrics handle zero values correctly."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        assert mse == 1.0
        assert rmse == 1.0
        assert mae == 1.0

