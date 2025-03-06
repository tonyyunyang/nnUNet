#!/usr/bin/env python3
import yaml
import subprocess
import os
import sys
from pathlib import Path


def run_commands(config_path):
    """Run the prediction and evaluation commands with the given config path."""
    # Run prediction script
    print(f"\nRunning prediction script with config: {config_path}")
    pred_result = subprocess.run(
        ["python3", "scripts/predict/predict.py", config_path],
        capture_output=True,
        text=True
    )
    if pred_result.returncode != 0:
        print(f"Prediction failed with error:\n{pred_result.stderr}")
        return False
    print(f"Prediction output:\n{pred_result.stdout}")

    # Run evaluation script
    print(f"\nRunning evaluation script with config: {config_path}")
    eval_result = subprocess.run(
        ["python3", "scripts/eval/eval.py", config_path],
        capture_output=True,
        text=True
    )
    if eval_result.returncode != 0:
        print(f"Evaluation failed with error:\n{eval_result.stderr}")
        return False
    print(f"Evaluation output:\n{eval_result.stdout}")

    return True


def update_config(config_path, fold, checkpoint_name):
    """Update the config file with the given fold and checkpoint name."""
    # Read the current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the config
    if 'predict' not in config:
        config['predict'] = {}

    config['predict']['fold'] = [fold]  # Note: keeping as a list with single element
    config['predict']['checkpoint_name'] = checkpoint_name

    # Write back the updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated config with fold=[{fold}], checkpoint_name={checkpoint_name}")
    return True


def main():
    # Define the config path
    config_path = "scripts/config/config.yaml"

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return 1

    # Create a backup of the original config
    backup_path = f"{config_path}.backup"
    with open(config_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    print(f"Created backup of original config at {backup_path}")

    # Process all combinations
    folds = [0, 1, 2, 3, 4]
    checkpoint_names = ["checkpoint_best.pth", "checkpoint_final.pth"]

    for fold in folds:
        for checkpoint_name in checkpoint_names:
            print(f"\n{'=' * 60}")
            print(f"Processing fold=[{fold}], checkpoint_name={checkpoint_name}")
            print(f"{'=' * 60}")

            # Update the config
            if not update_config(config_path, fold, checkpoint_name):
                continue

            # Run the commands
            run_commands(config_path)

    # Restore the original config
    with open(backup_path, 'r') as src, open(config_path, 'w') as dst:
        dst.write(src.read())
    print(f"\nRestored original config from {backup_path}")

    # Remove the backup
    os.remove(backup_path)
    print(f"Removed backup file {backup_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())