"""Derivative and distance computation for trajectory data."""

import numpy as np


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """Compute velocity as first-order derivative of positions.
    
    Velocity is computed as the difference between consecutive positions.
    The output has one fewer time step than the input.
    
    Args:
        positions: Array of positions with shape (n_steps, ...) or (n_steps, n_particles, dimension).
    
    Returns:
        Velocity array with shape (n_steps-1, ...) or (n_steps-1, n_particles, dimension).
    """
    return np.diff(positions, axis=0)


def compute_acceleration(positions: np.ndarray) -> np.ndarray:
    """Compute acceleration as second-order derivative of positions.
    
    Acceleration is computed as the difference of consecutive velocities.
    The output has two fewer time steps than the input.
    
    Args:
        positions: Array of positions with shape (n_steps, ...) or (n_steps, n_particles, dimension).
    
    Returns:
        Acceleration array with shape (n_steps-2, ...) or (n_steps-2, n_particles, dimension).
    """
    velocity = compute_velocity(positions)
    return np.diff(velocity, axis=0)


def compute_speed(positions: np.ndarray) -> np.ndarray:
    """Compute speed as the magnitude of velocity.
    
    Args:
        positions: Array of positions with shape (n_steps, dimension) or (n_steps, n_particles, dimension).
    
    Returns:
        Speed array with shape (n_steps-1,) or (n_steps-1, n_particles).
    """
    velocity = compute_velocity(positions)
    return np.linalg.norm(velocity, axis=-1)


def compute_displacement(positions: np.ndarray) -> np.ndarray:
    """Compute displacement from initial position.
    
    Args:
        positions: Array of positions with shape (n_steps, dimension) or (n_steps, n_particles, dimension).
    
    Returns:
        Displacement array with same shape as input.
    """
    return positions - positions[0]


def compute_distance_from_origin(positions: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from origin at each time step.
    
    Args:
        positions: Array of positions with shape (n_steps, dimension) or (n_steps, n_particles, dimension).
    
    Returns:
        Distance array with shape (n_steps,) or (n_steps, n_particles).
    """
    return np.linalg.norm(positions, axis=-1)


def compute_pairwise_distance(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between particles at each time step.
    
    Only applicable when n_particles > 1.
    
    Args:
        positions: Array of positions with shape (n_steps, n_particles, dimension).
    
    Returns:
        Pairwise distance matrix with shape (n_steps, n_particles, n_particles).
    """
    if positions.ndim != 3:
        raise ValueError("Pairwise distance requires 3D array (n_steps, n_particles, dimension)")
    
    n_steps, n_particles, dimension = positions.shape
    distances = np.zeros((n_steps, n_particles, n_particles))
    
    for t in range(n_steps):
        for i in range(n_particles):
            for j in range(n_particles):
                distances[t, i, j] = np.linalg.norm(positions[t, i] - positions[t, j])
    
    return distances

