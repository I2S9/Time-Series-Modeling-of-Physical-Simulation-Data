"""Script to generate simulated Brownian motion trajectories."""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.brownian import simulate_trajectories
from src.utils.seeding import set_seed


DEFAULT_N_STEPS = 50000
DEFAULT_NOISE_LEVEL = 0.1
DEFAULT_N_PARTICLES = 1
DEFAULT_DIMENSION = 2
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("data/raw")


def main():
    """Generate and save Brownian motion trajectories."""
    parser = argparse.ArgumentParser(
        description="Generate Brownian motion particle trajectories"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Number of time steps (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=DEFAULT_NOISE_LEVEL,
        help=f"Noise level (diffusion coefficient) (default: {DEFAULT_NOISE_LEVEL})",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=DEFAULT_N_PARTICLES,
        help=f"Number of particles (default: {DEFAULT_N_PARTICLES})",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        choices=[2, 3],
        help=f"Spatial dimension (default: {DEFAULT_DIMENSION})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated data (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="brownian_trajectories",
        help="Base name for output files (default: brownian_trajectories)",
    )
    parser.add_argument(
        "--drift",
        type=float,
        nargs="+",
        default=None,
        help="Drift vector components (default: None, no drift)",
    )
    parser.add_argument(
        "--nonlinear-interaction",
        action="store_true",
        help="Enable weak nonlinear interactions between particles",
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Generating trajectories with parameters:")
    print(f"  Time steps: {args.n_steps}")
    print(f"  Noise level: {args.noise_level}")
    print(f"  Particles: {args.n_particles}")
    print(f"  Dimension: {args.dimension}")
    print(f"  Seed: {args.seed}")
    
    drift = None
    if args.drift is not None:
        drift = np.array(args.drift)
        if len(drift) != args.dimension:
            raise ValueError(f"Drift must have {args.dimension} components, got {len(drift)}")
    
    trajectories = simulate_trajectories(
        n_steps=args.n_steps,
        noise_level=args.noise_level,
        n_particles=args.n_particles,
        dimension=args.dimension,
        drift=drift,
        nonlinear_interaction=args.nonlinear_interaction,
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path_npy = args.output_dir / f"{args.output_name}.npy"
    np.save(output_path_npy, trajectories)
    print(f"Saved trajectories to {output_path_npy}")
    print(f"  Shape: {trajectories.shape}")
    print(f"  Data type: {trajectories.dtype}")
    
    output_path_csv = args.output_dir / f"{args.output_name}.csv"
    np.savetxt(output_path_csv, trajectories, delimiter=",")
    print(f"Saved trajectories to {output_path_csv}")


if __name__ == "__main__":
    main()

