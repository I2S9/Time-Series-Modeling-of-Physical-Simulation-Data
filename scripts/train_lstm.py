"""Script to train an LSTM model on preprocessed trajectory data."""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm import LSTMModel
from src.training.lstm_trainer import LSTMTrainer
from src.evaluation.metrics import compute_metrics
from src.utils.seeding import set_seed


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "processed_trajectories"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_MODEL_DIR = Path("models/saved")
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_SEQ_LEN = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_N_EPOCHS = 100
DEFAULT_HORIZON = 1
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


def create_sequences(
    data: np.ndarray, seq_len: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Create input-output sequences from time series data.
    
    Args:
        data: Time series data with shape (n_samples, n_features).
        seq_len: Length of input sequences.
        horizon: Number of steps ahead to predict.
    
    Returns:
        Tuple of (X, y) where X has shape (n_sequences, seq_len, n_features)
        and y has shape (n_sequences, horizon, n_features).
    """
    n_samples, n_features = data.shape
    n_sequences = n_samples - seq_len - horizon + 1
    
    if n_sequences <= 0:
        raise ValueError(
            f"Not enough samples: need at least {seq_len + horizon}, got {n_samples}"
        )
    
    X = np.zeros((n_sequences, seq_len, n_features))
    y = np.zeros((n_sequences, horizon, n_features))
    
    for i in range(n_sequences):
        X[i] = data[i : i + seq_len]
        y[i] = data[i + seq_len : i + seq_len + horizon]
    
    return X, y


def create_dataloaders(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays.
    
    Args:
        train_X: Training input sequences.
        train_y: Training target sequences.
        val_X: Validation input sequences.
        val_y: Validation target sequences.
        test_X: Test input sequences.
        test_y: Test target sequences.
        batch_size: Batch size for training.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_y)
    )
    val_dataset = TensorDataset(torch.FloatTensor(val_X), torch.FloatTensor(val_y))
    test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_y))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Train an LSTM model on preprocessed data."""
    parser = argparse.ArgumentParser(description="Train an LSTM model")
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
        help=f"Directory to save trained model (default: {DEFAULT_MODEL_DIR})",
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
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"Hidden dimension of LSTM (default: {DEFAULT_HIDDEN_DIM})",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"Number of LSTM layers (default: {DEFAULT_NUM_LAYERS})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Input sequence length (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_N_EPOCHS})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Forecast horizon (default: {DEFAULT_HORIZON})",
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
    print(f"  Data type: {data.dtype}")
    
    train_data, val_data, test_data = split_data(
        data, args.train_split, args.val_split
    )
    print(f"\nData splits:")
    print(f"  Train: {train_data.shape[0]} samples")
    print(f"  Validation: {val_data.shape[0]} samples")
    print(f"  Test: {test_data.shape[0]} samples")
    
    print(f"\nCreating sequences...")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Forecast horizon: {args.horizon}")
    
    train_X, train_y = create_sequences(train_data, args.seq_len, args.horizon)
    val_X, val_y = create_sequences(val_data, args.seq_len, args.horizon)
    test_X, test_y = create_sequences(test_data, args.seq_len, args.horizon)
    
    print(f"  Train sequences: {train_X.shape[0]}")
    print(f"  Validation sequences: {val_X.shape[0]}")
    print(f"  Test sequences: {test_X.shape[0]}")
    
    input_dim = train_data.shape[1]
    print(f"\nModel configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.n_epochs}")
    
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_X, train_y, val_X, val_y, test_X, test_y, args.batch_size
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    trainer = LSTMTrainer(model=model, learning_rate=args.learning_rate, device=device)
    
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        verbose=True,
        print_every=10,
    )
    
    print("\nEvaluating on test set...")
    true_values, predictions, test_loss = trainer.evaluate_forecast(
        test_loader, horizon=args.horizon
    )
    
    if true_values is not None and predictions is not None:
        true_np = true_values.numpy()
        pred_np = predictions.numpy()
        
        true_flat = true_np.reshape(-1, true_np.shape[-1])
        pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
        
        metrics = compute_metrics(true_flat, pred_flat)
        
        print(f"\nTest set forecast metrics:")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
    else:
        metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0}
        print("Warning: Could not evaluate on test set")
    
    results = {
        "model_type": "lstm",
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "seq_len": int(args.seq_len),
        "horizon": int(args.horizon),
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_losses": [float(x) for x in history["train_loss"]],
        "val_losses": [float(x) for x in history["val_loss"]] if "val_loss" in history else None,
        "test_metrics": {
            "mse": float(metrics["mse"]),
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
        },
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]) if "val_loss" in history else None,
    }
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "lstm_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")
    
    baseline_results_path = args.output_dir / "baseline_results.json"
    if baseline_results_path.exists():
        print("\nComparing with baselines...")
        with open(baseline_results_path, "r") as f:
            baseline_results = json.load(f)
        
        print("\nTest set RMSE comparison:")
        print(f"  LSTM:           {metrics['rmse']:.6f}")
        if "persistence" in baseline_results and "test" in baseline_results["persistence"]:
            print(
                f"  Persistence:    {baseline_results['persistence']['test']['rmse']:.6f}"
            )
        if (
            "moving_average" in baseline_results
            and "test" in baseline_results["moving_average"]
        ):
            print(
                f"  Moving Average: {baseline_results['moving_average']['test']['rmse']:.6f}"
            )
        if (
            "linear_regression" in baseline_results
            and "test" in baseline_results["linear_regression"]
        ):
            print(
                f"  Linear Reg:     {baseline_results['linear_regression']['test']['rmse']:.6f}"
            )


if __name__ == "__main__":
    main()

