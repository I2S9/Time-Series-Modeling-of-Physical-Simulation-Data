"""Script to test autoencoder reconstruction."""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.autoencoder import Autoencoder


def main():
    """Test autoencoder reconstruction."""
    checkpoint_path = Path("models/saved/autoencoder.pt")
    test_data_path = Path("data/processed/processed_trajectories.npy")
    
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found: {checkpoint_path}")
        return
    
    if not test_data_path.exists():
        print(f"Test data not found: {test_data_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Autoencoder(
        input_dim=checkpoint["input_dim"], latent_dim=checkpoint["latent_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_data = np.load(test_data_path)[:100]
    test_tensor = torch.FloatTensor(test_data)
    
    with torch.no_grad():
        reconstructed, latent = model(test_tensor)
    
    print(f"Original shape: {test_data.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    print(
        f"Sample reconstruction MSE: {torch.mean((test_tensor - reconstructed)**2).item():.6f}"
    )
    print(f"Latent range: [{latent.min().item():.3f}, {latent.max().item():.3f}]")


if __name__ == "__main__":
    main()

