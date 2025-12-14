"""Evaluate models with rolling forecast to show horizon-dependent degradation."""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import (
    PersistenceModel,
    MovingAverageModel,
    LinearRegressionModel,
)
from src.evaluation.metrics import compute_metrics
from src.utils.seeding import set_seed


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "processed_trajectories"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_HORIZONS = [1, 2, 5, 10, 20, 50]
DEFAULT_WINDOW_SIZE = 10
DEFAULT_LOOKBACK_WINDOW = 10
DEFAULT_SEED = 42


def split_data(
    data: np.ndarray, train_ratio: float, val_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets."""
    n_samples = data.shape[0]
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def evaluate_rolling_forecast(
    model, train_data: np.ndarray, test_data: np.ndarray, horizon: int
) -> Dict[str, float]:
    """Evaluate model with rolling forecast (autoregressive multi-step).
    
    For horizon > 1, uses autoregressive prediction where each step
    uses the previous prediction, accumulating error over time.
    
    Args:
        model: Model instance.
        train_data: Training data for fitting.
        test_data: Test data for evaluation.
        horizon: Forecast horizon.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    model.fit(train_data)
    
    predictions = []
    true_values = []
    
    min_samples = 1
    if isinstance(model, LinearRegressionModel):
        min_samples = model.lookback_window
    
    for i in range(min_samples, len(test_data) - horizon):
        X = test_data[: i + 1]
        y_true = test_data[i + 1 : i + 1 + horizon]
        
        try:
            if horizon == 1:
                y_pred = model.predict(X, horizon=1)
            else:
                y_pred = model.predict(X, horizon=horizon)
            
            predictions.append(y_pred)
            true_values.append(y_true)
        except (ValueError, IndexError):
            continue
    
    if len(predictions) == 0:
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")}
    
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    
    metrics = compute_metrics(true_values, predictions)
    return metrics


def main():
    """Evaluate models with rolling forecast across horizons."""
    parser = argparse.ArgumentParser(
        description="Evaluate models with rolling forecast to show horizon degradation"
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
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help=f"Forecast horizons to test (default: {DEFAULT_HORIZONS})",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    input_path_npy = args.input_dir / f"{args.input_name}.npy"
    
    if not input_path_npy.exists():
        raise FileNotFoundError(f"Input file not found: {input_path_npy}")
    
    print("Rolling Forecast Evaluation")
    print("=" * 70)
    print(f"Forecast horizons: {args.horizons}")
    print("Using autoregressive multi-step forecasting")
    print("=" * 70)
    
    data = np.load(input_path_npy)
    train_data, val_data, test_data = split_data(
        data, 0.7, 0.15
    )
    
    all_results = {}
    
    for horizon in args.horizons:
        print(f"\nHorizon = {horizon}")
        print("-" * 70)
        
        horizon_results = {}
        
        persistence_model = PersistenceModel()
        persistence_metrics = evaluate_rolling_forecast(
            persistence_model, train_data, test_data, horizon
        )
        horizon_results["persistence"] = persistence_metrics
        print(f"  Persistence: RMSE = {persistence_metrics['rmse']:.6f}")
        
        moving_avg_model = MovingAverageModel(window_size=args.window_size)
        moving_avg_metrics = evaluate_rolling_forecast(
            moving_avg_model, train_data, test_data, horizon
        )
        horizon_results["moving_average"] = moving_avg_metrics
        print(f"  Moving Average: RMSE = {moving_avg_metrics['rmse']:.6f}")
        
        linear_reg_model = LinearRegressionModel(
            lookback_window=args.lookback_window
        )
        linear_reg_metrics = evaluate_rolling_forecast(
            linear_reg_model, train_data, test_data, horizon
        )
        horizon_results["linear_regression"] = linear_reg_metrics
        print(f"  Linear Regression: RMSE = {linear_reg_metrics['rmse']:.6f}")
        
        all_results[f"horizon_{horizon}"] = horizon_results
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "rolling_forecast_results.json"
    
    results_summary = {
        "horizons": args.horizons,
        "results": all_results,
        "method": "rolling_forecast_autoregressive",
    }
    
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nSaved results to {output_path}")
    
    print("\nSummary:")
    print(f"{'Horizon':<10} {'Persistence':<15} {'Moving Avg':<15} {'Linear Reg':<15}")
    print("-" * 70)
    for horizon in args.horizons:
        key = f"horizon_{horizon}"
        if key in all_results:
            pers = all_results[key]["persistence"]["rmse"]
            ma = all_results[key]["moving_average"]["rmse"]
            lr = all_results[key]["linear_regression"]["rmse"]
            print(f"{horizon:<10} {pers:<15.4f} {ma:<15.4f} {lr:<15.4f}")


if __name__ == "__main__":
    main()

