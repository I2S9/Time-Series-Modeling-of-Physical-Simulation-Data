"""Script to preprocess raw trajectory data and extract features."""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.preprocessing import extract_features


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_INPUT_NAME = "brownian_trajectories"


def main():
    """Preprocess trajectory data and extract features."""
    parser = argparse.ArgumentParser(
        description="Preprocess trajectory data and extract features"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory for raw data (default: {DEFAULT_INPUT_DIR})",
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
        help=f"Output directory for processed data (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="processed_trajectories",
        help="Base name for output files (default: processed_trajectories)",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        default=True,
        help="Include velocity features (default: True)",
    )
    parser.add_argument(
        "--no-velocity",
        dest="include_velocity",
        action="store_false",
        help="Exclude velocity features",
    )
    parser.add_argument(
        "--include-acceleration",
        action="store_true",
        help="Include acceleration features (default: False)",
    )
    parser.add_argument(
        "--include-speed",
        action="store_true",
        default=True,
        help="Include speed features (default: True)",
    )
    parser.add_argument(
        "--no-speed",
        dest="include_speed",
        action="store_false",
        help="Exclude speed features",
    )
    parser.add_argument(
        "--include-distance",
        action="store_true",
        default=True,
        help="Include distance from origin (default: True)",
    )
    parser.add_argument(
        "--no-distance",
        dest="include_distance",
        action="store_false",
        help="Exclude distance from origin",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize features (default: True)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do not normalize features",
    )
    parser.add_argument(
        "--normalization-method",
        type=str,
        default="zscore",
        choices=["zscore", "minmax"],
        help="Normalization method (default: zscore)",
    )
    
    args = parser.parse_args()
    
    input_path_npy = args.input_dir / f"{args.input_name}.npy"
    
    if not input_path_npy.exists():
        raise FileNotFoundError(f"Input file not found: {input_path_npy}")
    
    print(f"Loading data from {input_path_npy}")
    positions = np.load(input_path_npy)
    print(f"  Shape: {positions.shape}")
    print(f"  Data type: {positions.dtype}")
    
    print("\nExtracting features:")
    print(f"  Velocity: {args.include_velocity}")
    print(f"  Acceleration: {args.include_acceleration}")
    print(f"  Speed: {args.include_speed}")
    print(f"  Distance: {args.include_distance}")
    print(f"  Normalize: {args.normalize}")
    if args.normalize:
        print(f"  Normalization method: {args.normalization_method}")
    
    features, metadata = extract_features(
        positions,
        include_velocity=args.include_velocity,
        include_acceleration=args.include_acceleration,
        include_speed=args.include_speed,
        include_distance=args.include_distance,
        normalize=args.normalize,
        normalization_method=args.normalization_method,
    )
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Number of features: {metadata['n_features']}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path_npy = args.output_dir / f"{args.output_name}.npy"
    np.save(output_path_npy, features)
    print(f"\nSaved features to {output_path_npy}")
    
    metadata_path = args.output_dir / f"{args.output_name}_metadata.json"
    metadata_serializable = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            metadata_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            metadata_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            metadata_serializable[key] = value
    
    with open(metadata_path, "w") as f:
        json.dump(metadata_serializable, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    output_path_csv = args.output_dir / f"{args.output_name}.csv"
    np.savetxt(output_path_csv, features, delimiter=",")
    print(f"Saved features to {output_path_csv}")


if __name__ == "__main__":
    main()

