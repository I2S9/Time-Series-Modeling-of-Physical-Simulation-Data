"""Unit tests for simulation modules."""

import numpy as np
import pytest

from src.simulation.brownian import simulate_trajectories
from src.utils.seeding import set_seed


class TestBrownianSimulation:
    """Test cases for Brownian motion simulation."""
    
    def test_basic_trajectory_shape(self):
        """Test that trajectory has correct shape."""
        set_seed(42)
        positions = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        assert positions.shape == (100, 2)
        assert positions.dtype == np.float64
    
    def test_3d_trajectory(self):
        """Test 3D trajectory generation."""
        set_seed(42)
        positions = simulate_trajectories(n_steps=50, noise_level=0.1, dimension=3)
        
        assert positions.shape == (50, 3)
    
    def test_multiple_particles(self):
        """Test simulation with multiple particles."""
        set_seed(42)
        positions = simulate_trajectories(
            n_steps=100, noise_level=0.1, n_particles=3, dimension=2
        )
        
        assert positions.shape == (100, 3, 2)
    
    def test_single_particle_squeeze(self):
        """Test that single particle output is squeezed."""
        set_seed(42)
        positions = simulate_trajectories(
            n_steps=100, noise_level=0.1, n_particles=1, dimension=2
        )
        
        assert positions.shape == (100, 2)
        assert positions.ndim == 2
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        set_seed(42)
        positions1 = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        set_seed(42)
        positions2 = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        np.testing.assert_array_equal(positions1, positions2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        positions1 = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        set_seed(123)
        positions2 = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        assert not np.array_equal(positions1, positions2)
    
    def test_starts_at_origin(self):
        """Test that trajectory starts at origin."""
        set_seed(42)
        positions = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        np.testing.assert_array_equal(positions[0], [0.0, 0.0])
    
    def test_custom_initial_position(self):
        """Test trajectory with custom initial position."""
        set_seed(42)
        initial_pos = np.array([1.0, 2.0])
        positions = simulate_trajectories(
            n_steps=100, noise_level=0.1, dimension=2, initial_position=initial_pos
        )
        
        np.testing.assert_array_almost_equal(positions[0], initial_pos)
    
    def test_noise_level_affects_magnitude(self):
        """Test that higher noise level increases trajectory spread."""
        set_seed(42)
        positions_low = simulate_trajectories(n_steps=1000, noise_level=0.01, dimension=2)
        
        set_seed(42)
        positions_high = simulate_trajectories(n_steps=1000, noise_level=0.5, dimension=2)
        
        std_low = np.std(positions_low)
        std_high = np.std(positions_high)
        
        assert std_high > std_low
    
    def test_trajectory_continuity(self):
        """Test that trajectory is continuous (no jumps)."""
        set_seed(42)
        positions = simulate_trajectories(n_steps=100, noise_level=0.1, dimension=2)
        
        for i in range(1, len(positions)):
            diff = np.abs(positions[i] - positions[i - 1])
            assert np.all(diff < 1.0)

