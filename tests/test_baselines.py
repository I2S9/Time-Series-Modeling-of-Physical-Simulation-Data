"""Unit tests for baseline models."""

import numpy as np
import pytest

from src.models.baselines import (
    PersistenceModel,
    MovingAverageModel,
    LinearRegressionModel,
)


class TestPersistenceModel:
    """Test cases for Persistence model."""
    
    def test_fit_stores_last_value(self):
        """Test that fit stores the last value."""
        model = PersistenceModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        model.fit(X)
        
        np.testing.assert_array_equal(model.last_value, [5.0, 6.0])
    
    def test_predict_returns_last_value(self):
        """Test that predict returns the last observed value."""
        model = PersistenceModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        model.fit(X)
        predictions = model.predict(X, horizon=3)
        
        assert predictions.shape == (3, 2)
        np.testing.assert_array_equal(predictions[0], [5.0, 6.0])
        np.testing.assert_array_equal(predictions[1], [5.0, 6.0])
        np.testing.assert_array_equal(predictions[2], [5.0, 6.0])
    
    def test_predict_without_fit(self):
        """Test that predict works without explicit fit."""
        model = PersistenceModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        predictions = model.predict(X, horizon=1)
        
        assert predictions.shape == (1, 2)
        np.testing.assert_array_equal(predictions[0], [5.0, 6.0])
    
    def test_fit_empty_data_raises_error(self):
        """Test that fit on empty data raises error."""
        model = PersistenceModel()
        X = np.array([]).reshape(0, 2)
        
        with pytest.raises(ValueError):
            model.fit(X)


class TestMovingAverageModel:
    """Test cases for Moving Average model."""
    
    def test_fit_stores_history(self):
        """Test that fit stores recent history."""
        model = MovingAverageModel(window_size=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        model.fit(X)
        
        assert model.history.shape == (3, 1)
        np.testing.assert_array_equal(model.history, [[3.0], [4.0], [5.0]])
    
    def test_predict_returns_average(self):
        """Test that predict returns the average of window."""
        model = MovingAverageModel(window_size=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        model.fit(X)
        predictions = model.predict(X, horizon=2)
        
        expected_mean = np.mean([3.0, 4.0, 5.0])
        assert predictions.shape == (2, 1)
        np.testing.assert_array_almost_equal(predictions[0], [expected_mean])
        np.testing.assert_array_almost_equal(predictions[1], [expected_mean])
    
    def test_window_size_parameter(self):
        """Test that window size parameter works correctly."""
        model = MovingAverageModel(window_size=2)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        model.fit(X)
        predictions = model.predict(X, horizon=1)
        
        expected_mean = np.mean([4.0, 5.0])
        np.testing.assert_array_almost_equal(predictions[0], [expected_mean])
    
    def test_invalid_window_size_raises_error(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError):
            MovingAverageModel(window_size=0)
    
    def test_predict_without_fit(self):
        """Test that predict works without explicit fit."""
        model = MovingAverageModel(window_size=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        predictions = model.predict(X, horizon=1)
        
        expected_mean = np.mean([3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(predictions[0], [expected_mean])


class TestLinearRegressionModel:
    """Test cases for Linear Regression model."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict functionality."""
        model = LinearRegressionModel(lookback_window=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
        
        model.fit(X)
        predictions = model.predict(X, horizon=1)
        
        assert predictions.shape == (1, 1)
        assert not np.isnan(predictions[0, 0])
    
    def test_lookback_window_parameter(self):
        """Test that lookback window parameter works."""
        model = LinearRegressionModel(lookback_window=5)
        X = np.random.randn(20, 2)
        
        model.fit(X)
        predictions = model.predict(X, horizon=1)
        
        assert predictions.shape == (1, 2)
    
    def test_invalid_lookback_window_raises_error(self):
        """Test that invalid lookback window raises error."""
        with pytest.raises(ValueError):
            LinearRegressionModel(lookback_window=0)
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises error."""
        model = LinearRegressionModel(lookback_window=10)
        X = np.array([[1.0], [2.0], [3.0]])
        
        with pytest.raises(ValueError):
            model.fit(X)
    
    def test_predict_requires_fit(self):
        """Test that predict requires model to be fitted."""
        model = LinearRegressionModel(lookback_window=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        with pytest.raises(ValueError):
            model.predict(X, horizon=1)
    
    def test_multistep_prediction(self):
        """Test multi-step ahead prediction."""
        model = LinearRegressionModel(lookback_window=3)
        X = np.random.randn(20, 2)
        
        model.fit(X)
        predictions = model.predict(X, horizon=5)
        
        assert predictions.shape == (5, 2)
        assert not np.any(np.isnan(predictions))

