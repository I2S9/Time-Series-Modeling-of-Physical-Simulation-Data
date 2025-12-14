"""Unit tests for feature extraction and preprocessing."""

import numpy as np
import pytest

from src.features.derivatives import (
    compute_velocity,
    compute_acceleration,
    compute_speed,
    compute_distance_from_origin,
)
from src.features.normalization import (
    normalize_minmax,
    normalize_zscore,
    denormalize_minmax,
    denormalize_zscore,
)


class TestDerivatives:
    """Test cases for derivative computations."""
    
    def test_velocity_shape(self):
        """Test that velocity has correct shape."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        velocity = compute_velocity(positions)
        
        assert velocity.shape == (3, 2)
    
    def test_velocity_computation(self):
        """Test that velocity is computed correctly."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        velocity = compute_velocity(positions)
        
        np.testing.assert_array_equal(velocity[0], [1.0, 1.0])
        np.testing.assert_array_equal(velocity[1], [1.0, 1.0])
    
    def test_acceleration_shape(self):
        """Test that acceleration has correct shape."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        acceleration = compute_acceleration(positions)
        
        assert acceleration.shape == (3, 2)
    
    def test_speed_computation(self):
        """Test that speed is computed correctly."""
        positions = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        speed = compute_speed(positions)
        
        expected_speed = np.sqrt(3**2 + 4**2)
        np.testing.assert_array_almost_equal(speed[0], expected_speed)
    
    def test_distance_from_origin(self):
        """Test distance from origin computation."""
        positions = np.array([[0.0, 0.0], [3.0, 4.0], [5.0, 0.0]])
        distances = compute_distance_from_origin(positions)
        
        np.testing.assert_array_almost_equal(distances[0], 0.0)
        np.testing.assert_array_almost_equal(distances[1], 5.0)
        np.testing.assert_array_almost_equal(distances[2], 5.0)


class TestNormalization:
    """Test cases for normalization functions."""
    
    def test_minmax_normalization_range(self):
        """Test that minmax normalization produces values in [0, 1]."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, _, _ = normalize_minmax(data)
        
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
    
    def test_minmax_denormalization(self):
        """Test that denormalization recovers original data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, data_min, data_max = normalize_minmax(data)
        denormalized = denormalize_minmax(normalized, data_min, data_max)
        
        np.testing.assert_array_almost_equal(denormalized, data)
    
    def test_zscore_normalization_mean_zero(self):
        """Test that zscore normalization produces zero mean."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, _, _ = normalize_zscore(data)
        
        mean = np.mean(normalized, axis=0)
        np.testing.assert_array_almost_equal(mean, [0.0, 0.0])
    
    def test_zscore_normalization_std_one(self):
        """Test that zscore normalization produces unit std."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, _, _ = normalize_zscore(data)
        
        std = np.std(normalized, axis=0)
        np.testing.assert_array_almost_equal(std, [1.0, 1.0])
    
    def test_zscore_denormalization(self):
        """Test that zscore denormalization recovers original data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, data_mean, data_std = normalize_zscore(data)
        denormalized = denormalize_zscore(normalized, data_mean, data_std)
        
        np.testing.assert_array_almost_equal(denormalized, data)
    
    def test_normalization_handles_constant_data(self):
        """Test that normalization handles constant data."""
        data = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        
        normalized_minmax, _, _ = normalize_minmax(data)
        normalized_zscore, _, _ = normalize_zscore(data)
        
        assert not np.any(np.isnan(normalized_minmax))
        assert not np.any(np.isnan(normalized_zscore))

