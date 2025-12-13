"""Training utilities for LSTM models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional


class LSTMTrainer:
    """Trainer for LSTM models."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: LSTM model to train.
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
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            output, _ = self.model(x)
            
            if y.shape[1] == 1:
                output = output[:, -1:, :]
            
            loss = self.criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                output, _ = self.model(x)
                
                if y.shape[1] == 1:
                    output = output[:, -1:, :]
                
                loss = self.criterion(output, y)
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
    
    def evaluate_forecast(
        self, dataloader: DataLoader, horizon: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Evaluate forecasting performance.
        
        Args:
            dataloader: DataLoader for evaluation data.
            horizon: Number of steps ahead to forecast.
        
        Returns:
            Tuple of (true_values, predictions, mse_loss).
        """
        self.model.eval()
        all_true = []
        all_pred = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                y_true = batch[1].to(self.device)
                
                predictions = self.model.predict_step(x, horizon=horizon)
                
                if y_true.shape[1] == horizon:
                    loss = self.criterion(predictions, y_true)
                    all_true.append(y_true.cpu())
                    all_pred.append(predictions.cpu())
                    total_loss += loss.item()
                    n_batches += 1
        
        if len(all_true) == 0:
            return None, None, 0.0
        
        true_values = torch.cat(all_true, dim=0)
        predictions = torch.cat(all_pred, dim=0)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        
        return true_values, predictions, avg_loss

