"""Script to train an autoencoder on preprocessed trajectory data."""

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

from src.models.autoencoder import Autoencoder, DeepAutoencoder
from src.training.trainer import AutoencoderTrainer
from src.evaluation.metrics import compute_metrics
from src.utils.seeding import set_seed


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "processed_trajectories"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_MODEL_DIR = Path("models/saved")
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_LATENT_DIM = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_N_EPOCHS = 100
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


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays.
    
    Args:
        train_data: Training data.
        val_data: Validation data.
        test_data: Test data.
        batch_size: Batch size for training.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = TensorDataset(torch.FloatTensor(train_data))
    val_dataset = TensorDataset(torch.FloatTensor(val_data))
    test_dataset = TensorDataset(torch.FloatTensor(test_data))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Train an autoencoder on preprocessed data."""
    parser = argparse.ArgumentParser(description="Train an autoencoder model")
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
        "--latent-dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
        help=f"Latent dimension (default: {DEFAULT_LATENT_DIM})",
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use deep autoencoder instead of simple linear autoencoder",
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
    
    input_dim = train_data.shape[1]
    print(f"\nModel configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Model type: {'Deep' if args.deep else 'Simple'} autoencoder")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.n_epochs}")
    
    if args.deep:
        model = DeepAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim)
    else:
        model = Autoencoder(input_dim=input_dim, latent_dim=args.latent_dim)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, args.batch_size
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    trainer = AutoencoderTrainer(
        model=model, learning_rate=args.learning_rate, device=device
    )
    
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        verbose=True,
        print_every=10,
    )
    
    print("\nEvaluating on test set...")
    original, reconstructed, test_loss = trainer.evaluate_reconstruction(test_loader)
    
    original_np = original.numpy()
    reconstructed_np = reconstructed.numpy()
    
    metrics = compute_metrics(original_np, reconstructed_np)
    
    print(f"\nTest set reconstruction metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    
    results = {
        "model_type": "deep" if args.deep else "simple",
        "input_dim": int(input_dim),
        "latent_dim": int(args.latent_dim),
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
    output_path = args.output_dir / "autoencoder_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": args.latent_dim,
            "model_type": "deep" if args.deep else "simple",
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

