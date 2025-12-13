"""Script to evaluate classical baseline models on preprocessed data."""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import (
    PersistenceModel,
    MovingAverageModel,
    LinearRegressionModel,
)
from src.evaluation.metrics import compute_metrics


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "processed_trajectories"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_HORIZON = 1
DEFAULT_WINDOW_SIZE = 10
DEFAULT_LOOKBACK_WINDOW = 10


def split_data(
    data: np.ndarray, train_ratio: float, val_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.
    
    Args:
        data: Time series data with shape (n_samples, n_features).
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
    
    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    n_samples = data.shape[0]
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def evaluate_model(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    horizon: int = 1,
    min_samples: int = 1,
) -> dict:
    """Evaluate a model on train, validation, and test sets.
    
    Args:
        model: Model instance with fit() and predict() methods.
        train_data: Training data.
        val_data: Validation data.
        test_data: Test data.
        horizon: Number of steps ahead to predict.
        min_samples: Minimum number of samples required for prediction.
    
    Returns:
        Dictionary containing metrics for each split.
    """
    model.fit(train_data)
    
    results = {}
    
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        if len(split_data) < horizon + min_samples:
            continue
        
        predictions = []
        true_values = []
        
        for i in range(min_samples, len(split_data) - horizon):
            X = split_data[: i + 1]
            y_true = split_data[i + 1 : i + 1 + horizon]
            
            try:
                y_pred = model.predict(X, horizon=horizon)
                predictions.append(y_pred)
                true_values.append(y_true)
            except (ValueError, IndexError):
                continue
        
        if len(predictions) == 0:
            continue
        
        predictions = np.concatenate(predictions, axis=0)
        true_values = np.concatenate(true_values, axis=0)
        
        metrics = compute_metrics(true_values, predictions)
        results[split_name] = metrics
    
    return results


def main():
    """Evaluate baseline models on preprocessed data."""
    parser = argparse.ArgumentParser(
        description="Evaluate classical baseline models"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory for processed data (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default=DEFAULT_INPUT_NAME,
        help=f"Base name of input file without extension (default: {DEFAULT_INPUT_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"Proportion of data for training (default: {DEFAULT_TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help=f"Proportion of data for validation (default: {DEFAULT_VAL_SPLIT})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Forecast horizon (number of steps ahead) (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Window size for moving average (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=DEFAULT_LOOKBACK_WINDOW,
        help=f"Lookback window for linear regression (default: {DEFAULT_LOOKBACK_WINDOW})",
    )
    
    args = parser.parse_args()
    
    input_path_npy = args.input_dir / f"{args.input_name}.npy"
    
    if not input_path_npy.exists():
        raise FileNotFoundError(f"Input file not found: {input_path_npy}")
    
    print(f"Loading data from {input_path_npy}")
    data = np.load(input_path_npy)
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    
    train_data, val_data, test_data = split_data(
        data, args.train_split, args.val_split
    )
    print(f"\nData splits:")
    print(f"  Train: {train_data.shape[0]} samples")
    print(f"  Validation: {val_data.shape[0]} samples")
    print(f"  Test: {test_data.shape[0]} samples")
    print(f"  Forecast horizon: {args.horizon} steps")
    
    models = {
        "persistence": PersistenceModel(),
        "moving_average": MovingAverageModel(window_size=args.window_size),
        "linear_regression": LinearRegressionModel(
            lookback_window=args.lookback_window
        ),
    }
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        min_samples = 1
        if model_name == "linear_regression":
            min_samples = args.lookback_window
        results = evaluate_model(
            model,
            train_data,
            val_data,
            test_data,
            horizon=args.horizon,
            min_samples=min_samples,
        )
        all_results[model_name] = results
        
        for split_name, metrics in results.items():
            print(f"  {split_name}:")
            print(f"    MSE:  {metrics['mse']:.6f}")
            print(f"    RMSE: {metrics['rmse']:.6f}")
            print(f"    MAE:  {metrics['mae']:.6f}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output_dir / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    print("\nSummary (Test set RMSE):")
    for model_name, results in all_results.items():
        if "test" in results:
            print(f"  {model_name}: {results['test']['rmse']:.6f}")


if __name__ == "__main__":
    main()

