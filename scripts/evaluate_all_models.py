"""Systematic evaluation script comparing all models on multiple forecast horizons."""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import (
    PersistenceModel,
    MovingAverageModel,
    LinearRegressionModel,
)
from src.models.lstm import LSTMModel
from src.evaluation.metrics import compute_metrics
from src.utils.seeding import set_seed


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "processed_trajectories"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_MODEL_DIR = Path("models/saved")
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_SEQ_LEN = 10
DEFAULT_WINDOW_SIZE = 10
DEFAULT_LOOKBACK_WINDOW = 10
DEFAULT_HORIZONS = [1, 5, 10]
DEFAULT_SEED = 42


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


def evaluate_baseline_model(
    model, train_data: np.ndarray, test_data: np.ndarray, horizon: int
) -> Dict[str, float]:
    """Evaluate a baseline model on test data.
    
    Args:
        model: Baseline model instance.
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
            if horizon > 1:
                y_pred = model.predict(X, horizon=horizon)
            else:
                y_pred = model.predict(X, horizon=1)
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


def evaluate_lstm_model(
    model_path: Path,
    test_data: np.ndarray,
    seq_len: int,
    horizon: int,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate LSTM model on test data.
    
    Args:
        model_path: Path to saved LSTM model.
        test_data: Test data for evaluation.
        seq_len: Input sequence length.
        horizon: Forecast horizon.
        batch_size: Batch size for evaluation.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    if not model_path.exists():
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")}
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    n_samples, n_features = test_data.shape
    n_sequences = n_samples - seq_len - horizon + 1
    
    if n_sequences <= 0:
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")}
    
    X = np.zeros((n_sequences, seq_len, n_features))
    y_true = np.zeros((n_sequences, horizon, n_features))
    
    for i in range(n_sequences):
        X[i] = test_data[i : i + seq_len]
        y_true[i] = test_data[i + seq_len : i + seq_len + horizon]
    
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        predictions = model.predict_step(X_tensor, horizon=horizon)
    
    predictions_np = predictions.numpy()
    
    true_flat = y_true.reshape(-1, y_true.shape[-1])
    pred_flat = predictions_np.reshape(-1, predictions_np.shape[-1])
    
    metrics = compute_metrics(true_flat, pred_flat)
    return metrics


def main():
    """Systematically evaluate all models on multiple horizons."""
    parser = argparse.ArgumentParser(
        description="Systematically evaluate all models on multiple forecast horizons"
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
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory with saved models (default: {DEFAULT_MODEL_DIR})",
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
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Sequence length for LSTM (default: {DEFAULT_SEQ_LEN})",
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
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help=f"Forecast horizons to evaluate (default: {DEFAULT_HORIZONS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    input_path_npy = args.input_dir / f"{args.input_name}.npy"
    
    if not input_path_npy.exists():
        raise FileNotFoundError(f"Input file not found: {input_path_npy}")
    
    print(f"Loading data from {input_path_npy}")
    data = np.load(input_path_npy)
    print(f"  Shape: {data.shape}")
    
    train_data, val_data, test_data = split_data(
        data, args.train_split, args.val_split
    )
    print(f"\nData splits:")
    print(f"  Train: {train_data.shape[0]} samples")
    print(f"  Validation: {val_data.shape[0]} samples")
    print(f"  Test: {test_data.shape[0]} samples")
    print(f"\nEvaluating on horizons: {args.horizons}")
    
    all_results = {}
    baseline_rmse = {}
    
    for horizon in args.horizons:
        print(f"\n{'='*60}")
        print(f"Evaluating horizon = {horizon}")
        print(f"{'='*60}")
        
        horizon_results = {}
        
        print("\nBaseline models:")
        persistence_model = PersistenceModel()
        persistence_metrics = evaluate_baseline_model(
            persistence_model, train_data, test_data, horizon
        )
        horizon_results["persistence"] = persistence_metrics
        print(f"  Persistence: RMSE = {persistence_metrics['rmse']:.6f}")
        if horizon == args.horizons[0]:
            baseline_rmse["persistence"] = persistence_metrics["rmse"]
        
        moving_avg_model = MovingAverageModel(window_size=args.window_size)
        moving_avg_metrics = evaluate_baseline_model(
            moving_avg_model, train_data, test_data, horizon
        )
        horizon_results["moving_average"] = moving_avg_metrics
        print(f"  Moving Average: RMSE = {moving_avg_metrics['rmse']:.6f}")
        if horizon == args.horizons[0]:
            baseline_rmse["moving_average"] = moving_avg_metrics["rmse"]
        
        linear_reg_model = LinearRegressionModel(
            lookback_window=args.lookback_window
        )
        linear_reg_metrics = evaluate_baseline_model(
            linear_reg_model, train_data, test_data, horizon
        )
        horizon_results["linear_regression"] = linear_reg_metrics
        print(f"  Linear Regression: RMSE = {linear_reg_metrics['rmse']:.6f}")
        if horizon == args.horizons[0]:
            baseline_rmse["linear_regression"] = linear_reg_metrics["rmse"]
        
        print("\nDeep learning models:")
        lstm_model_path = args.model_dir / "lstm.pt"
        lstm_metrics = evaluate_lstm_model(
            lstm_model_path, test_data, args.seq_len, horizon
        )
        horizon_results["lstm"] = lstm_metrics
        if lstm_metrics["rmse"] != float("inf"):
            print(f"  LSTM: RMSE = {lstm_metrics['rmse']:.6f}")
            
            if horizon == args.horizons[0]:
                best_baseline_rmse = min(baseline_rmse.values())
                improvement = (
                    (best_baseline_rmse - lstm_metrics["rmse"])
                    / best_baseline_rmse
                    * 100
                )
                print(f"  Improvement vs best baseline: {improvement:.2f}%")
        else:
            print(f"  LSTM: Model not found or evaluation failed")
        
        all_results[f"horizon_{horizon}"] = horizon_results
    
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    
    print("\nRMSE by Model and Horizon:")
    print(f"{'Model':<20} " + " ".join([f"H={h:>3}" for h in args.horizons]))
    print("-" * (20 + 4 * len(args.horizons)))
    
    model_names = ["persistence", "moving_average", "linear_regression", "lstm"]
    for model_name in model_names:
        row = f"{model_name:<20} "
        for horizon in args.horizons:
            key = f"horizon_{horizon}"
            if key in all_results and model_name in all_results[key]:
                rmse = all_results[key][model_name]["rmse"]
                if rmse != float("inf"):
                    row += f"{rmse:>7.4f} "
                else:
                    row += f"{'N/A':>7} "
            else:
                row += f"{'N/A':>7} "
        print(row)
    
    print("\nMSE by Model and Horizon:")
    print(f"{'Model':<20} " + " ".join([f"H={h:>3}" for h in args.horizons]))
    print("-" * (20 + 4 * len(args.horizons)))
    
    for model_name in model_names:
        row = f"{model_name:<20} "
        for horizon in args.horizons:
            key = f"horizon_{horizon}"
            if key in all_results and model_name in all_results[key]:
                mse = all_results[key][model_name]["mse"]
                if mse != float("inf"):
                    row += f"{mse:>7.4f} "
                else:
                    row += f"{'N/A':>7} "
            else:
                row += f"{'N/A':>7} "
        print(row)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "systematic_evaluation.json"
    
    results_summary = {
        "horizons": args.horizons,
        "results": all_results,
        "baseline_rmse_horizon_1": baseline_rmse,
    }
    
    if "lstm" in all_results.get(f"horizon_{args.horizons[0]}", {}):
        lstm_rmse_h1 = all_results[f"horizon_{args.horizons[0]}"]["lstm"]["rmse"]
        if lstm_rmse_h1 != float("inf"):
            best_baseline_rmse = min(baseline_rmse.values())
            improvement = (best_baseline_rmse - lstm_rmse_h1) / best_baseline_rmse * 100
            results_summary["improvement_vs_best_baseline_percent"] = float(improvement)
    
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    if "improvement_vs_best_baseline_percent" in results_summary:
        improvement = results_summary["improvement_vs_best_baseline_percent"]
        print(f"\nLSTM improvement vs best baseline (H=1): {improvement:.2f}%")


if __name__ == "__main__":
    main()

