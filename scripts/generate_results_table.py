"""Generate formatted results table from systematic evaluation."""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_FILE = Path("reports/systematic_evaluation.json")
DEFAULT_OUTPUT_FILE = Path("reports/results_table.md")


def main():
    """Generate formatted results table."""
    parser = argparse.ArgumentParser(description="Generate formatted results table")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSON file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output markdown file (default: {DEFAULT_OUTPUT_FILE})",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return
    
    with open(args.input, "r") as f:
        data = json.load(f)
    
    horizons = data["horizons"]
    results = data["results"]
    
    output_lines = []
    output_lines.append("# Model Evaluation Results\n")
    output_lines.append("## RMSE by Model and Forecast Horizon\n\n")
    output_lines.append("| Model | " + " | ".join([f"H={h}" for h in horizons]) + " |")
    output_lines.append("|-------|" + "|".join(["-------" for _ in horizons]) + "|")
    
    model_names = ["persistence", "moving_average", "linear_regression", "lstm"]
    model_display_names = {
        "persistence": "Persistence",
        "moving_average": "Moving Average",
        "linear_regression": "Linear Regression",
        "lstm": "LSTM",
    }
    
    for model_name in model_names:
        row = [model_display_names.get(model_name, model_name)]
        for horizon in horizons:
            key = f"horizon_{horizon}"
            if key in results and model_name in results[key]:
                rmse = results[key][model_name]["rmse"]
                if rmse != float("inf"):
                    row.append(f"{rmse:.4f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        output_lines.append("| " + " | ".join(row) + " |")
    
    output_lines.append("\n## MSE by Model and Forecast Horizon\n\n")
    output_lines.append("| Model | " + " | ".join([f"H={h}" for h in horizons]) + " |")
    output_lines.append("|-------|" + "|".join(["-------" for _ in horizons]) + "|")
    
    for model_name in model_names:
        row = [model_display_names.get(model_name, model_name)]
        for horizon in horizons:
            key = f"horizon_{horizon}"
            if key in results and model_name in results[key]:
                mse = results[key][model_name]["mse"]
                if mse != float("inf"):
                    row.append(f"{mse:.4f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        output_lines.append("| " + " | ".join(row) + " |")
    
    if "improvement_vs_best_baseline_percent" in data:
        improvement = data["improvement_vs_best_baseline_percent"]
        output_lines.append(f"\n## Improvement\n\n")
        output_lines.append(
            f"LSTM improvement vs best baseline (H=1): {improvement:.2f}%\n"
        )
    
    baseline_rmse = data.get("baseline_rmse_horizon_1", {})
    if baseline_rmse and "lstm" in results.get("horizon_1", {}):
        lstm_rmse = results["horizon_1"]["lstm"]["rmse"]
        if lstm_rmse != float("inf"):
            best_baseline = min(baseline_rmse.values())
            output_lines.append(f"\nBest baseline RMSE (H=1): {best_baseline:.4f}\n")
            output_lines.append(f"LSTM RMSE (H=1): {lstm_rmse:.4f}\n")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"Generated results table: {args.output}")
    print("\n" + "\n".join(output_lines))


if __name__ == "__main__":
    main()

