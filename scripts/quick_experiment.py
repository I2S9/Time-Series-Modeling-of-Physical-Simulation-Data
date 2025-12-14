"""Quick experiment script with minimal configuration."""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run a quick experiment with default settings."""
    print("=" * 70)
    print("QUICK EXPERIMENT - Using default configuration")
    print("=" * 70)
    
    config_path = Path("experiments/default_config.json")
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please run from project root directory.")
        return 1
    
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config", str(config_path),
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

