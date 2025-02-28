#!/usr/bin/env python3
import yaml
import subprocess
import sys
import os
import glob
from pathlib import Path


def read_config(yaml_file):
    """Read configuration from YAML file."""
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{yaml_file}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def find_dataset_folder(dataset_id, base_dir="nnUNet_results"):
    """Find folder containing the given dataset ID."""
    # If nnUNet_results is an environment variable, use it
    if "nnUNet_results" in os.environ:
        base_dir = os.environ["nnUNet_results"]

    # Ensure the base directory exists
    full_base_dir = os.path.abspath(base_dir)
    if not os.path.exists(full_base_dir):
        print(f"Error: Base directory {full_base_dir} does not exist")
        return None

    # Format the dataset ID with leading zeros to ensure it's 3 digits
    dataset_id_str = f"{int(dataset_id):03d}"

    # Look for folders matching Dataset{dataset_id}*
    pattern = os.path.join(full_base_dir, f"Dataset{dataset_id_str}*")
    matching_dirs = glob.glob(pattern)

    if not matching_dirs:
        print(f"Error: No directory matching pattern {pattern} found")
        return None

    # Return the first matching directory
    print(f"Found dataset directory: {matching_dirs[0]}")
    return matching_dirs[0]


def find_model_folder(dataset_folder, trainer="FlexibleTrainerV1", plans="nnUNetPlans", configuration="2d"):
    """Find the model folder within the dataset folder."""
    if not dataset_folder or not os.path.exists(dataset_folder):
        return None

    # Try exact match first
    pattern = os.path.join(dataset_folder, f"{trainer}__{plans}__{configuration}")
    if os.path.exists(pattern):
        return pattern

    # If exact match not found, try a more flexible pattern
    pattern = os.path.join(dataset_folder, f"{trainer}*{plans}*{configuration}")
    matching_dirs = glob.glob(pattern)

    if not matching_dirs:
        print(f"Error: No model directory found in {dataset_folder}")
        return None

    # Return the first matching directory
    print(f"Found model directory: {matching_dirs[0]}")
    return matching_dirs[0]


def build_evaluate_command(gt_folder, pred_folder, dataset_json, plans_json, output_file=None, num_processes=None,
                           chill=False):
    """Build the nnUNetv2_evaluate command."""
    cmd = ['nnUNetv2_evaluate_folder']

    # Add required arguments
    cmd.append(gt_folder)
    cmd.append(pred_folder)
    cmd.extend(['-djfile', dataset_json])
    cmd.extend(['-pfile', plans_json])

    # Add optional arguments
    if output_file is not None:
        cmd.extend(['-o', output_file])

    if num_processes is not None:
        cmd.extend(['-np', str(num_processes)])

    if chill:
        cmd.append('--chill')

    return cmd


def execute_command(cmd):
    """Execute the command and stream the output to the console."""
    cmd_str = ' '.join(cmd)
    print(f"Executing command: {cmd_str}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        print("Make sure nnUNetv2 is properly installed and in your PATH")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 run_evaluation.py config.yaml")
        sys.exit(1)

    yaml_file = sys.argv[1]
    config = read_config(yaml_file)

    # Get evaluation configuration
    if 'evaluate' not in config:
        print("Error: 'evaluate' section not found in the config file.")
        sys.exit(1)

    evaluate_config = config['evaluate']
    predict_config = config.get('predict', {})
    train_config = config.get('train', {})

    # Check for required parameters
    if 'gt_folder' not in evaluate_config:
        print("Error: 'gt_folder' is required in the evaluate config")
        sys.exit(1)

    # For prediction folder, use the one from the prediction config if available
    prediction_folder = predict_config.get('output_folder')

    # If prediction_folder is not specified in the config, derive it from evaluate_config
    if prediction_folder is None and 'pred_folder' in evaluate_config:
        prediction_folder = evaluate_config['pred_folder']

    if prediction_folder is None:
        print("Error: prediction folder not specified in config")
        sys.exit(1)

    # Check if single fold was used, which means we need to adjust the prediction folder path
    folds = predict_config.get('fold', train_config.get('fold', [0, 1, 2, 3, 4]))
    if not isinstance(folds, list):
        folds = [folds]

    if len(folds) == 1:
        # Prediction results are in a subdirectory for single fold
        fold_subdir = os.path.join(prediction_folder, f"fold_{folds[0]}")
        if os.path.exists(fold_subdir):
            prediction_folder = fold_subdir
            print(f"Using predictions from single fold directory: {prediction_folder}")

    # Get dataset_id from training config
    dataset_id = None
    if 'dataset_name_or_id' in train_config:
        dataset_id = train_config['dataset_name_or_id']

    # If model_folder is specified directly, use it
    model_folder = predict_config.get('model_folder')

    if not model_folder and dataset_id:
        # Find dataset folder based on dataset_id
        dataset_folder = find_dataset_folder(dataset_id)
        if dataset_folder:
            # Get trainer, plans, and configuration from config or use defaults
            trainer = train_config.get('tr', 'FlexibleTrainerV1')
            plans = train_config.get('p', 'nnUNetPlans')
            configuration = train_config.get('configuration', '2d')

            # Find model folder
            model_folder = find_model_folder(dataset_folder, trainer, plans, configuration)

    if not model_folder:
        print(f"Error: Could not find model folder.")
        sys.exit(1)

    print(f"Using model folder: {model_folder}")

    # Find dataset.json and plans.json
    dataset_json = os.path.join(model_folder, "dataset.json")
    plans_json = os.path.join(model_folder, "plans.json")

    if not os.path.exists(dataset_json):
        print(f"Error: dataset.json not found at {dataset_json}")
        sys.exit(1)

    if not os.path.exists(plans_json):
        print(f"Error: plans.json not found at {plans_json}")
        sys.exit(1)

    # Set output file if provided
    output_file = evaluate_config.get('output_file')

    # Get number of processes
    num_processes = evaluate_config.get('num_processes')

    # Get chill option
    chill = evaluate_config.get('chill', False)

    # Build and execute the command
    cmd = build_evaluate_command(
        evaluate_config['gt_folder'],
        prediction_folder,
        dataset_json,
        plans_json,
        output_file,
        num_processes,
        chill
    )

    execute_command(cmd)

    # Determine the output file path (either specified or default)
    if output_file is None:
        output_file = os.path.join(prediction_folder, 'summary.json')

    print(f"Evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()