"""Main script to run a complete reproducible experiment."""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: Path) -> Dict:
    """Load experiment configuration from JSON file.
    
    Args:
        config_path: Path to configuration file.
    
    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def run_command(command: list, description: str) -> bool:
    """Run a shell command and handle errors.
    
    Args:
        command: List of command and arguments.
        description: Description of what the command does.
    
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed with error: {e}")
        return False


def main():
    """Run complete experiment pipeline."""
    parser = argparse.ArgumentParser(
        description="Run complete reproducible experiment pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/default_config.json"),
        help="Path to configuration file (default: experiments/default_config.json)",
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation step (use existing data)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step (use existing processed data)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training steps (use existing models)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation steps",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["generate", "preprocess", "baselines", "autoencoder", "lstm", "evaluate", "robustness", "failure"],
        help="Run only specific steps (default: all)",
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    
    print("=" * 70)
    print("REPRODUCIBLE EXPERIMENT PIPELINE")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Project root: {project_root}")
    print("=" * 70)
    
    steps_to_run = []
    
    if args.steps:
        steps_to_run = args.steps
    else:
        if not args.skip_data_generation:
            steps_to_run.append("generate")
        if not args.skip_preprocessing:
            steps_to_run.append("preprocess")
        if not args.skip_training:
            steps_to_run.extend(["baselines", "autoencoder", "lstm"])
        if not args.skip_evaluation:
            steps_to_run.extend(["evaluate"])
            if config.get("robustness", {}).get("enabled", False):
                steps_to_run.append("robustness")
            if config.get("failure_analysis", {}).get("enabled", False):
                steps_to_run.append("failure")
    
    print(f"\nSteps to run: {', '.join(steps_to_run)}")
    
    success = True
    
    if "generate" in steps_to_run:
        data_config = config["data_generation"]
        cmd = [
            "python",
            "scripts/generate_data.py",
            "--n-steps", str(data_config["n_steps"]),
            "--noise-level", str(data_config["noise_level"]),
            "--n-particles", str(data_config["n_particles"]),
            "--dimension", str(data_config["dimension"]),
            "--seed", str(data_config["seed"]),
            "--output-name", data_config["output_name"],
        ]
        success = run_command(cmd, "Data Generation") and success
    
    if "preprocess" in steps_to_run:
        prep_config = config["preprocessing"]
        cmd = [
            "python",
            "scripts/preprocess_data.py",
            "--input-name", config["data_generation"]["output_name"],
            "--output-name", prep_config["output_name"],
        ]
        if prep_config.get("include_velocity", True):
            cmd.append("--include-velocity")
        else:
            cmd.append("--no-velocity")
        if prep_config.get("include_acceleration", False):
            cmd.append("--include-acceleration")
        if prep_config.get("include_speed", True):
            cmd.append("--include-speed")
        else:
            cmd.append("--no-speed")
        if prep_config.get("include_distance", True):
            cmd.append("--include-distance")
        else:
            cmd.append("--no-distance")
        if prep_config.get("normalize", True):
            cmd.append("--normalize")
        else:
            cmd.append("--no-normalize")
        cmd.extend(["--normalization-method", prep_config.get("normalization_method", "zscore")])
        
        success = run_command(cmd, "Data Preprocessing") and success
    
    if "baselines" in steps_to_run:
        baseline_config = config["baselines"]
        cmd = [
            "python",
            "scripts/evaluate_baselines.py",
            "--window-size", str(baseline_config["window_size"]),
            "--lookback-window", str(baseline_config["lookback_window"]),
            "--horizon", str(baseline_config["horizon"]),
        ]
        success = run_command(cmd, "Baseline Evaluation") and success
    
    if "autoencoder" in steps_to_run:
        ae_config = config["autoencoder"]
        cmd = [
            "python",
            "scripts/train_autoencoder.py",
            "--latent-dim", str(ae_config["latent_dim"]),
            "--batch-size", str(ae_config["batch_size"]),
            "--learning-rate", str(ae_config["learning_rate"]),
            "--n-epochs", str(ae_config["n_epochs"]),
        ]
        if ae_config.get("deep", False):
            cmd.append("--deep")
        success = run_command(cmd, "Autoencoder Training") and success
    
    if "lstm" in steps_to_run:
        lstm_config = config["lstm"]
        cmd = [
            "python",
            "scripts/train_lstm.py",
            "--hidden-dim", str(lstm_config["hidden_dim"]),
            "--num-layers", str(lstm_config["num_layers"]),
            "--seq-len", str(lstm_config["seq_len"]),
            "--batch-size", str(lstm_config["batch_size"]),
            "--learning-rate", str(lstm_config["learning_rate"]),
            "--n-epochs", str(lstm_config["n_epochs"]),
        ]
        success = run_command(cmd, "LSTM Training") and success
    
    if "evaluate" in steps_to_run:
        eval_config = config["evaluation"]
        horizons = eval_config.get("horizons", [1, 5, 10])
        cmd = [
            "python",
            "scripts/evaluate_all_models.py",
            "--horizons",
        ] + [str(h) for h in horizons]
        success = run_command(cmd, "Model Evaluation") and success
        
        cmd = [
            "python",
            "scripts/generate_results_table.py",
        ]
        success = run_command(cmd, "Results Table Generation") and success
        
        cmd = [
            "python",
            "scripts/generate_report.py",
        ]
        success = run_command(cmd, "Comprehensive Report Generation") and success
    
    if "robustness" in steps_to_run:
        robust_config = config["robustness"]
        cmd = [
            "python",
            "scripts/robustness_study.py",
            "--n-steps", str(robust_config["n_steps"]),
            "--noise-levels",
        ] + [str(nl) for nl in robust_config["noise_levels"]] + [
            "--seeds",
        ] + [str(s) for s in robust_config["seeds"]] + [
            "--n-epochs", str(robust_config["n_epochs"]),
        ]
        success = run_command(cmd, "Robustness Study") and success
    
    if "failure" in steps_to_run:
        failure_config = config["failure_analysis"]
        horizons = failure_config.get("horizons", [1, 2, 5, 10, 20])
        cmd = [
            "python",
            "scripts/analyze_failures.py",
            "--horizons",
        ] + [str(h) for h in horizons]
        success = run_command(cmd, "Failure Analysis") and success
    
    if "report" in steps_to_run:
        cmd = [
            "python",
            "scripts/generate_report.py",
        ]
        success = run_command(cmd, "Comprehensive Report Generation") and success
    
    print(f"\n{'='*70}")
    if success:
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nResults are available in:")
        print("  - reports/ : Evaluation results and visualizations")
        print("  - models/saved/ : Trained models")
        print("  - data/processed/ : Preprocessed data")
        return 0
    else:
        print("EXPERIMENT COMPLETED WITH ERRORS")
        print("=" * 70)
        print("Check the output above for details on which steps failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

