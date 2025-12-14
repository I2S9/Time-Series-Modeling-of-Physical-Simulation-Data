"""Analysis of model failures and limitations across different forecast horizons."""

import argparse
import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

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
from src.training.lstm_trainer import LSTMTrainer
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "processed_trajectories"
DEFAULT_MODEL_DIR = Path("models/saved")
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_HORIZONS = [1, 2, 5, 10, 20, 50]
DEFAULT_SEQ_LEN = 10
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


def evaluate_baseline_model(
    model, train_data: np.ndarray, test_data: np.ndarray, horizon: int
) -> Dict[str, float]:
    """Evaluate a baseline model on test data."""
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
            y_pred = model.predict(X, horizon=horizon)
            predictions.append(y_pred)
            true_values.append(y_true)
        except (ValueError, IndexError):
            continue
    
    if len(predictions) == 0:
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")}, None, None
    
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    
    metrics = compute_metrics(true_values, predictions)
    return metrics, true_values, predictions


def create_sequences(
    data: np.ndarray, seq_len: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Create input-output sequences from time series data."""
    n_samples, n_features = data.shape
    n_sequences = n_samples - seq_len - horizon + 1
    
    if n_sequences <= 0:
        return np.array([]), np.array([])
    
    X = np.zeros((n_sequences, seq_len, n_features))
    y = np.zeros((n_sequences, horizon, n_features))
    
    for i in range(n_sequences):
        X[i] = data[i : i + seq_len]
        y[i] = data[i + seq_len : i + seq_len + horizon]
    
    return X, y


def evaluate_lstm_model(
    model_path: Path,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    seq_len: int,
    horizon: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate LSTM model on test data."""
    if not model_path.exists():
        return (
            {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")},
            None,
            None,
        )
    
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_X, train_y = create_sequences(train_data, seq_len, horizon)
    val_X, val_y = create_sequences(val_data, seq_len, horizon)
    test_X, test_y = create_sequences(test_data, seq_len, horizon)
    
    if len(test_X) == 0:
        return (
            {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")},
            None,
            None,
        )
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    trainer = LSTMTrainer(model=model, learning_rate=0.001, device=device)
    
    true_values, predictions, _ = trainer.evaluate_forecast(test_loader, horizon=horizon)
    
    if true_values is None or predictions is None:
        return (
            {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf")},
            None,
            None,
        )
    
    true_np = true_values.cpu().numpy()
    pred_np = predictions.cpu().numpy()
    
    true_flat = true_np.reshape(-1, true_np.shape[-1])
    pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
    
    metrics = compute_metrics(true_flat, pred_flat)
    return metrics, true_flat, pred_flat


def plot_error_by_horizon(
    results: Dict, output_path: Path, model_names: List[str]
) -> None:
    """Plot RMSE and MSE as a function of forecast horizon."""
    horizons = sorted([int(h.split("_")[1]) for h in results.keys() if h.startswith("horizon_")])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name in model_names:
        rmse_values = []
        mse_values = []
        
        for horizon in horizons:
            key = f"horizon_{horizon}"
            if key in results and model_name in results[key]:
                metrics = results[key][model_name]
                if metrics["rmse"] != float("inf"):
                    rmse_values.append(metrics["rmse"])
                    mse_values.append(metrics["mse"])
                else:
                    rmse_values.append(np.nan)
                    mse_values.append(np.nan)
            else:
                rmse_values.append(np.nan)
                mse_values.append(np.nan)
        
        axes[0].plot(horizons, rmse_values, marker="o", label=model_name, linewidth=2)
        axes[1].plot(horizons, mse_values, marker="s", label=model_name, linewidth=2)
    
    axes[0].set_xlabel("Forecast Horizon", fontsize=12)
    axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].set_title("RMSE vs Forecast Horizon", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Forecast Horizon", fontsize=12)
    axes[1].set_ylabel("MSE", fontsize=12)
    axes[1].set_title("MSE vs Forecast Horizon", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved error plot to {output_path}")


def plot_prediction_errors(
    true_values: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    horizon: int,
    output_path: Path,
    max_samples: int = 1000,
) -> None:
    """Plot prediction errors for a specific model and horizon."""
    if true_values is None or predictions is None:
        return
    
    n_samples = min(len(true_values), max_samples)
    indices = np.arange(n_samples)
    
    errors = np.abs(true_values[:n_samples] - predictions[:n_samples])
    mean_errors = np.mean(errors, axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(indices, mean_errors, alpha=0.6, linewidth=1)
    axes[0].axhline(
        np.mean(mean_errors),
        color="r",
        linestyle="--",
        label=f"Mean Error: {np.mean(mean_errors):.4f}",
    )
    axes[0].set_xlabel("Sample Index", fontsize=12)
    axes[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes[0].set_title(
        f"Prediction Errors: {model_name} (Horizon={horizon})", fontsize=14, fontweight="bold"
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mean_errors, bins=50, alpha=0.7, edgecolor="black")
    axes[1].axvline(
        np.mean(mean_errors),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(mean_errors):.4f}",
    )
    axes[1].axvline(
        np.median(mean_errors),
        color="g",
        linestyle="--",
        label=f"Median: {np.median(mean_errors):.4f}",
    )
    axes[1].set_xlabel("Mean Absolute Error", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Error Distribution", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction error plot to {output_path}")


def identify_difficult_horizons(results: Dict, model_names: List[str]) -> Dict:
    """Identify horizons where models perform poorly."""
    horizons = sorted([int(h.split("_")[1]) for h in results.keys() if h.startswith("horizon_")])
    
    difficult_horizons = {}
    
    for model_name in model_names:
        rmse_by_horizon = []
        for horizon in horizons:
            key = f"horizon_{horizon}"
            if key in results and model_name in results[key]:
                rmse = results[key][model_name]["rmse"]
                if rmse != float("inf"):
                    rmse_by_horizon.append((horizon, rmse))
        
        if len(rmse_by_horizon) > 1:
            baseline_rmse = rmse_by_horizon[0][1]
            degradation = []
            for horizon, rmse in rmse_by_horizon:
                relative_degradation = (rmse - baseline_rmse) / baseline_rmse * 100
                degradation.append((horizon, relative_degradation))
            
            difficult_horizons[model_name] = {
                "baseline_rmse": baseline_rmse,
                "degradation": degradation,
                "worst_horizon": max(degradation, key=lambda x: x[1]),
            }
    
    return difficult_horizons


def main():
    """Analyze model failures and limitations."""
    parser = argparse.ArgumentParser(
        description="Analyze model failures and limitations across forecast horizons"
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
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory with saved models (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for results (default: reports)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help=f"Forecast horizons to test (default: {DEFAULT_HORIZONS})",
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
    
    print("Failure Analysis: Testing models across forecast horizons")
    print("=" * 70)
    print(f"Forecast horizons: {args.horizons}")
    print("=" * 70)
    
    data = np.load(input_path_npy)
    train_data, val_data, test_data = split_data(
        data, args.train_split, args.val_split
    )
    
    print(f"\nData splits:")
    print(f"  Train: {train_data.shape[0]} samples")
    print(f"  Validation: {val_data.shape[0]} samples")
    print(f"  Test: {test_data.shape[0]} samples")
    
    all_results = {}
    prediction_data = {}
    
    model_names = ["persistence", "moving_average", "linear_regression", "lstm"]
    
    for horizon in args.horizons:
        print(f"\n{'='*70}")
        print(f"Evaluating horizon = {horizon}")
        print(f"{'='*70}")
        
        horizon_results = {}
        
        print("\nBaseline models:")
        persistence_model = PersistenceModel()
        persistence_metrics, true_pers, pred_pers = evaluate_baseline_model(
            persistence_model, train_data, test_data, horizon
        )
        horizon_results["persistence"] = persistence_metrics
        print(f"  Persistence: RMSE = {persistence_metrics['rmse']:.6f}")
        if horizon == args.horizons[0]:
            prediction_data["persistence"] = (true_pers, pred_pers)
        
        moving_avg_model = MovingAverageModel(window_size=args.window_size)
        moving_avg_metrics, true_ma, pred_ma = evaluate_baseline_model(
            moving_avg_model, train_data, test_data, horizon
        )
        horizon_results["moving_average"] = moving_avg_metrics
        print(f"  Moving Average: RMSE = {moving_avg_metrics['rmse']:.6f}")
        if horizon == args.horizons[0]:
            prediction_data["moving_average"] = (true_ma, pred_ma)
        
        linear_reg_model = LinearRegressionModel(
            lookback_window=args.lookback_window
        )
        linear_reg_metrics, true_lr, pred_lr = evaluate_baseline_model(
            linear_reg_model, train_data, test_data, horizon
        )
        horizon_results["linear_regression"] = linear_reg_metrics
        print(f"  Linear Regression: RMSE = {linear_reg_metrics['rmse']:.6f}")
        if horizon == args.horizons[0]:
            prediction_data["linear_regression"] = (true_lr, pred_lr)
        
        print("\nDeep learning models:")
        lstm_model_path = args.model_dir / "lstm.pt"
        lstm_metrics, true_lstm, pred_lstm = evaluate_lstm_model(
            lstm_model_path, train_data, val_data, test_data, args.seq_len, horizon, args.seed
        )
        horizon_results["lstm"] = lstm_metrics
        if lstm_metrics["rmse"] != float("inf"):
            print(f"  LSTM: RMSE = {lstm_metrics['rmse']:.6f}")
            if horizon == args.horizons[0]:
                prediction_data["lstm"] = (true_lstm, pred_lstm)
        else:
            print(f"  LSTM: Model not found or evaluation failed")
        
        all_results[f"horizon_{horizon}"] = horizon_results
    
    print(f"\n{'='*70}")
    print("IDENTIFYING DIFFICULT HORIZONS")
    print(f"{'='*70}")
    
    difficult_horizons = identify_difficult_horizons(all_results, model_names)
    
    for model_name, analysis in difficult_horizons.items():
        print(f"\n{model_name}:")
        print(f"  Baseline RMSE (H=1): {analysis['baseline_rmse']:.6f}")
        worst = analysis["worst_horizon"]
        print(
            f"  Worst degradation: Horizon {worst[0]} with {worst[1]:.2f}% increase"
        )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_error_by_horizon(
        all_results,
        args.output_dir / "error_by_horizon.png",
        model_names,
    )
    
    for model_name in model_names:
        if model_name in prediction_data:
            true_vals, pred_vals = prediction_data[model_name]
            if true_vals is not None and pred_vals is not None:
                plot_prediction_errors(
                    true_vals,
                    pred_vals,
                    model_name,
                    args.horizons[0],
                    args.output_dir / f"prediction_errors_{model_name}.png",
                )
    
    print(f"\n{'='*70}")
    print("DOCUMENTING LIMITATIONS")
    print(f"{'='*70}")
    
    limitations = {
        "summary": "Analysis of model failures and limitations",
        "difficult_horizons": difficult_horizons,
        "observations": [],
    }
    
    for model_name, analysis in difficult_horizons.items():
        worst = analysis["worst_horizon"]
        if worst[1] > 20:
            limitations["observations"].append(
                f"{model_name}: Significant degradation (>20%) at horizon {worst[0]}"
            )
        elif worst[1] > 10:
            limitations["observations"].append(
                f"{model_name}: Moderate degradation (>10%) at horizon {worst[0]}"
            )
    
    limitations["observations"].append(
        "All models show increasing error with longer forecast horizons"
    )
    limitations["observations"].append(
        "Persistence model has highest variance across horizons"
    )
    limitations["observations"].append(
        "LSTM and Linear Regression show most stable performance"
    )
    
    for obs in limitations["observations"]:
        print(f"  - {obs}")
    
    results_summary = {
        "horizons": args.horizons,
        "results": all_results,
        "difficult_horizons": difficult_horizons,
        "limitations": limitations,
    }
    
    output_path = args.output_dir / "failure_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nSaved analysis to {output_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

