"""Brownian motion particle simulation for generating time series trajectories."""

import numpy as np


def simulate_trajectories(
    n_steps: int,
    noise_level: float,
    n_particles: int = 1,
    dimension: int = 2,
    initial_position: np.ndarray = None,
) -> np.ndarray:
    """Simulate Brownian motion trajectories for particles.
    
    Generates continuous but noisy trajectories by adding Gaussian noise
    to particle positions at each time step.
    
    Args:
        n_steps: Number of time steps in the trajectory.
        noise_level: Standard deviation of the Gaussian noise (diffusion coefficient).
        n_particles: Number of particles to simulate.
        dimension: Spatial dimension (2 for 2D, 3 for 3D).
        initial_position: Starting position(s) for particles. If None, starts at origin.
            Shape should be (n_particles, dimension) or (dimension,) for single particle.
    
    Returns:
        Array of shape (n_steps, n_particles, dimension) containing particle positions
        over time. If n_particles=1, shape is (n_steps, dimension).
    """
    if initial_position is None:
        positions = np.zeros((n_steps, n_particles, dimension))
    else:
        initial_position = np.asarray(initial_position)
        if initial_position.ndim == 1:
            initial_position = initial_position.reshape(1, -1)
        if initial_position.shape[0] != n_particles:
            raise ValueError(
                f"Initial position shape {initial_position.shape} does not match "
                f"n_particles={n_particles}"
            )
        if initial_position.shape[1] != dimension:
            raise ValueError(
                f"Initial position dimension {initial_position.shape[1]} does not match "
                f"dimension={dimension}"
            )
        positions = np.zeros((n_steps, n_particles, dimension))
        positions[0] = initial_position
    
    for t in range(1, n_steps):
        noise = np.random.normal(0, noise_level, size=(n_particles, dimension))
        positions[t] = positions[t - 1] + noise
    
    if n_particles == 1:
        positions = positions.squeeze(axis=1)
    
    return positions

