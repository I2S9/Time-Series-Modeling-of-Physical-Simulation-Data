"""LSTM model for temporal sequence modeling and forecasting."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting.
    
    Uses LSTM layers to model temporal dependencies in sequential data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        output_dim: int = None,
        dropout: float = 0.0,
    ):
        """Initialize the LSTM model.
        
        Args:
            input_dim: Dimension of input features at each time step.
            hidden_dim: Dimension of hidden state in LSTM.
            num_layers: Number of stacked LSTM layers.
            output_dim: Dimension of output. If None, uses input_dim.
            dropout: Dropout probability (applied between LSTM layers if num_layers > 1).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.fc = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(
        self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the LSTM.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, input_dim).
            hidden: Optional tuple of (h_n, c_n) hidden states.
        
        Returns:
            Tuple of (output, (h_n, c_n)) where output has shape
            (batch_size, seq_len, output_dim) and (h_n, c_n) are final hidden states.
        """
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden
    
    def predict_step(
        self, x: torch.Tensor, horizon: int = 1
    ) -> torch.Tensor:
        """Predict future steps autoregressively.
        
        Args:
            x: Input sequence with shape (batch_size, seq_len, input_dim).
            horizon: Number of future steps to predict.
        
        Returns:
            Predictions with shape (batch_size, horizon, output_dim).
        """
        self.eval()
        batch_size = x.shape[0]
        device = x.device
        
        with torch.no_grad():
            _, (h_n, c_n) = self.lstm(x)
            
            predictions = []
            last_output = self.fc(self.lstm(x)[0][:, -1:, :])
            
            for _ in range(horizon):
                lstm_out, (h_n, c_n) = self.lstm(last_output, (h_n, c_n))
                pred = self.fc(lstm_out)
                predictions.append(pred)
                last_output = pred
            
            return torch.cat(predictions, dim=1)


class BidirectionalLSTMModel(nn.Module):
    """Bidirectional LSTM model for time series forecasting.
    
    Uses bidirectional LSTM to capture both forward and backward temporal dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        output_dim: int = None,
        dropout: float = 0.0,
    ):
        """Initialize the bidirectional LSTM model.
        
        Args:
            input_dim: Dimension of input features at each time step.
            hidden_dim: Dimension of hidden state in LSTM.
            num_layers: Number of stacked LSTM layers.
            output_dim: Dimension of output. If None, uses input_dim.
            dropout: Dropout probability (applied between LSTM layers if num_layers > 1).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.fc = nn.Linear(hidden_dim * 2, self.output_dim)
    
    def forward(
        self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the bidirectional LSTM.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, input_dim).
            hidden: Optional tuple of (h_n, c_n) hidden states.
        
        Returns:
            Tuple of (output, (h_n, c_n)) where output has shape
            (batch_size, seq_len, output_dim) and (h_n, c_n) are final hidden states.
        """
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden

