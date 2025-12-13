"""Training utilities for neural network models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Optional


class AutoencoderTrainer:
    """Trainer for autoencoder models."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: Autoencoder model to train.
            learning_rate: Learning rate for optimizer.
            device: Device to run training on (CPU or CUDA).
        """
        self.model = model
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data.
        
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(x)
            loss = self.criterion(reconstructed, x)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model.
        
        Args:
            dataloader: DataLoader for validation data.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstructed, _ = self.model(x)
                loss = self.criterion(reconstructed, x)
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100,
        verbose: bool = True,
        print_every: int = 10,
    ) -> dict:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            n_epochs: Number of training epochs.
            verbose: Whether to print training progress.
            print_every: Print progress every N epochs.
        
        Returns:
            Dictionary containing training history.
        """
        best_val_loss = float("inf")
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            else:
                val_loss = None
            
            if verbose and (epoch + 1) % print_every == 0:
                if val_loss is not None:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.6f}"
                    )
        
        history = {"train_loss": self.train_losses}
        if val_loader is not None:
            history["val_loss"] = self.val_losses
        
        return history
    
    def evaluate_reconstruction(
        self, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Evaluate reconstruction quality.
        
        Args:
            dataloader: DataLoader for evaluation data.
        
        Returns:
            Tuple of (original_data, reconstructed_data, mse_loss).
        """
        self.model.eval()
        all_original = []
        all_reconstructed = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstructed, _ = self.model(x)
                loss = self.criterion(reconstructed, x)
                
                all_original.append(x.cpu())
                all_reconstructed.append(reconstructed.cpu())
                total_loss += loss.item()
                n_batches += 1
        
        original = torch.cat(all_original, dim=0)
        reconstructed = torch.cat(all_reconstructed, dim=0)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        
        return original, reconstructed, avg_loss

