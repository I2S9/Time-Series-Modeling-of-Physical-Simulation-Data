# Experiment Configurations

This directory contains configuration files for running reproducible experiments.

## Default Configuration

`default_config.json` contains the default experiment settings. It includes:

- **Data Generation**: Parameters for generating Brownian motion trajectories
- **Preprocessing**: Feature extraction and normalization options
- **Training**: Hyperparameters for autoencoder and LSTM models
- **Evaluation**: Forecast horizons and evaluation settings
- **Robustness**: Optional robustness study settings
- **Failure Analysis**: Optional failure analysis settings

## Running Experiments

### Using Default Configuration

```bash
python scripts/run_experiment.py
```

### Using Custom Configuration

1. Create a new JSON file in this directory (e.g., `my_experiment.json`)
2. Copy the structure from `default_config.json`
3. Modify parameters as needed
4. Run:

```bash
python scripts/run_experiment.py --config experiments/my_experiment.json
```

### Running Specific Steps

Run only certain steps:

```bash
# Generate and preprocess data only
python scripts/run_experiment.py --steps generate preprocess

# Train models only
python scripts/run_experiment.py --steps baselines autoencoder lstm

# Evaluate only
python scripts/run_experiment.py --steps evaluate
```

### Skipping Steps

Skip certain steps (useful when data/models already exist):

```bash
# Skip data generation and preprocessing
python scripts/run_experiment.py --skip-data-generation --skip-preprocessing

# Skip training (use existing models)
python scripts/run_experiment.py --skip-training
```

## Configuration Structure

Each configuration file should follow this structure:

```json
{
  "data_generation": {
    "n_steps": 50000,
    "noise_level": 0.1,
    "n_particles": 1,
    "dimension": 2,
    "seed": 42,
    "output_name": "brownian_trajectories"
  },
  "preprocessing": {
    "include_velocity": true,
    "include_acceleration": false,
    "include_speed": true,
    "include_distance": true,
    "normalize": true,
    "normalization_method": "zscore",
    "output_name": "processed_trajectories"
  },
  "training": {
    "train_split": 0.7,
    "val_split": 0.15,
    "seed": 42
  },
  "baselines": {
    "window_size": 10,
    "lookback_window": 10,
    "horizon": 1
  },
  "autoencoder": {
    "latent_dim": 2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "n_epochs": 100,
    "deep": false
  },
  "lstm": {
    "hidden_dim": 64,
    "num_layers": 1,
    "seq_len": 10,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "n_epochs": 30
  },
  "evaluation": {
    "horizons": [1, 5, 10],
    "generate_plots": true
  },
  "robustness": {
    "enabled": false,
    "noise_levels": [0.05, 0.1, 0.15, 0.2],
    "seeds": [42, 123, 456],
    "n_steps": 20000,
    "n_epochs": 20
  },
  "failure_analysis": {
    "enabled": false,
    "horizons": [1, 2, 5, 10, 20]
  }
}
```

## Reproducibility

All experiments use fixed random seeds specified in the configuration. To reproduce an experiment:

1. Use the same configuration file
2. Ensure the same Python version and dependencies
3. Run the experiment from the project root directory

Results will be identical across runs with the same configuration.

