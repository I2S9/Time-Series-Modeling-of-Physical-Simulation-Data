# Test Suite

This directory contains unit tests for the time series modeling project.

## Running Tests

To run all tests:
```bash
pytest tests/ -v
```

To run a specific test file:
```bash
pytest tests/test_simulation.py -v
```

To run a specific test:
```bash
pytest tests/test_simulation.py::TestBrownianSimulation::test_basic_trajectory_shape -v
```

## Test Coverage

### Simulation Tests (`test_simulation.py`)
- Basic trajectory generation (2D, 3D, multiple particles)
- Reproducibility with seeds
- Custom initial positions
- Noise level effects
- Trajectory continuity

### Metrics Tests (`test_metrics.py`)
- MSE, RMSE, MAE computation
- Perfect predictions
- Multidimensional arrays
- Edge cases (zeros, constant errors)
- Metric consistency

### Baseline Model Tests (`test_baselines.py`)
- Persistence model
- Moving Average model
- Linear Regression model
- Fit and predict functionality
- Error handling

### Feature Extraction Tests (`test_features.py`)
- Derivative computations (velocity, acceleration, speed)
- Distance calculations
- Normalization (minmax, zscore)
- Denormalization
- Edge cases

## Test Principles

All tests follow these principles:
- **Deterministic**: Tests use fixed seeds for reproducibility
- **Isolated**: Each test is independent
- **Fast**: Tests run quickly without heavy computation
- **Clear**: Test names describe what is being tested

