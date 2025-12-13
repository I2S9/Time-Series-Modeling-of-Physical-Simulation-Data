"""Autoencoder model for dimensionality reduction and noise filtering."""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple autoencoder with linear encoder and decoder.
    
    The autoencoder learns a compressed representation (latent space) of the input
    and reconstructs the original input from this representation.
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        """Initialize the autoencoder.
        
        Args:
            input_dim: Dimension of the input features.
            latent_dim: Dimension of the latent representation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim).
        
        Returns:
            Tuple of (reconstructed, latent) tensors.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim).
        
        Returns:
            Latent representation with shape (batch_size, latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor with shape (batch_size, latent_dim).
        
        Returns:
            Reconstructed input with shape (batch_size, input_dim).
        """
        return self.decoder(z)


class DeepAutoencoder(nn.Module):
    """Deep autoencoder with multiple hidden layers.
    
    Provides more capacity for learning complex representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = None,
        activation: nn.Module = nn.ReLU(),
    ):
        """Initialize the deep autoencoder.
        
        Args:
            input_dim: Dimension of the input features.
            latent_dim: Dimension of the latent representation.
            hidden_dims: List of hidden layer dimensions for encoder/decoder.
                If None, uses [64, 32] as default.
            activation: Activation function to use between layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation)
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(activation)
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim).
        
        Returns:
            Tuple of (reconstructed, latent) tensors.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim).
        
        Returns:
            Latent representation with shape (batch_size, latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor with shape (batch_size, latent_dim).
        
        Returns:
            Reconstructed input with shape (batch_size, input_dim).
        """
        return self.decoder(z)

