# Scripts Directory

This directory contains all executable scripts for the project.

## Main Experiment Script

### `run_experiment.py`

Main script to run complete reproducible experiments. Orchestrates the entire pipeline from data generation to evaluation.

**Usage:**
```bash
python scripts/run_experiment.py [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file (default: `experiments/default_config.json`)
- `--skip-data-generation`: Skip data generation step
- `--skip-preprocessing`: Skip preprocessing step
- `--skip-training`: Skip training steps
- `--skip-evaluation`: Skip evaluation steps
- `--steps STEP1 STEP2 ...`: Run only specific steps

**Example:**
```bash
# Run complete experiment
python scripts/run_experiment.py

# Run with custom config
python scripts/run_experiment.py --config experiments/my_config.json

# Run only data generation and preprocessing
python scripts/run_experiment.py --steps generate preprocess
```

## Individual Scripts

### Data Generation
- `generate_data.py`: Generate Brownian motion trajectories

### Preprocessing
- `preprocess_data.py`: Extract features and normalize data

### Training
- `train_autoencoder.py`: Train autoencoder model
- `train_lstm.py`: Train LSTM model

### Evaluation
- `evaluate_baselines.py`: Evaluate baseline models
- `evaluate_all_models.py`: Systematic evaluation of all models
- `generate_results_table.py`: Generate formatted results table

### Analysis
- `robustness_study.py`: Robustness analysis across noise levels
- `analyze_failures.py`: Failure mode analysis

## Quick Start

For a quick experiment with default settings:

```bash
python scripts/quick_experiment.py
```

Or use the main script:

```bash
python scripts/run_experiment.py
```

## Script Execution Order

The typical execution order for a complete experiment:

1. `generate_data.py` - Generate raw trajectory data
2. `preprocess_data.py` - Extract features and normalize
3. `evaluate_baselines.py` - Evaluate baseline models
4. `train_autoencoder.py` - Train autoencoder
5. `train_lstm.py` - Train LSTM
6. `evaluate_all_models.py` - Compare all models
7. `generate_results_table.py` - Format results
8. (Optional) `robustness_study.py` - Robustness analysis
9. (Optional) `analyze_failures.py` - Failure analysis

The `run_experiment.py` script automates this entire pipeline.

