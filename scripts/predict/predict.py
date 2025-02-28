#!/usr/bin/env python3
import yaml
import subprocess
import sys
import os
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


def build_predict_command(config, train_config):
    """Build the predict command based on configuration."""
    # Start with the base command, which depends on whether we're using model_folder or dataset approach
    if 'model_folder' in config:
        cmd = ['nnUNetv2_predict_from_modelfolder']
    else:
        cmd = ['nnUNetv2_predict']

    # Fold handling - get this first as we may need to modify the output path
    if 'fold' in config:
        folds = config['fold']
    else:
        folds = train_config['fold']

    if not isinstance(folds, list):
        folds = [folds]

    # Check if only a single fold is specified
    output_folder = str(config['output_folder'])
    if len(folds) == 1:
        # Create a subfolder for this specific fold
        fold_subdir = os.path.join(output_folder, f"fold_{folds[0]}")
        os.makedirs(fold_subdir, exist_ok=True)
        print(f"Using single fold {folds[0]}, saving results to {fold_subdir}")
        # Update the output folder to the fold-specific subdirectory
        output_folder = fold_subdir

    # Required arguments
    cmd.extend(['-i', str(config['input_folder'])])
    cmd.extend(['-o', output_folder])

    # Check which style of command we need to build
    if 'model_folder' in config:
        # We're using the model_folder method (predict_entry_point_modelfolder)
        cmd.extend(['-m', str(config['model_folder'])])
    else:
        # We're using the dataset method (predict_entry_point)
        cmd.extend(['-d', str(train_config['dataset_name_or_id'])])

        if 'configuration' in train_config:
            cmd.extend(['-c', str(train_config['configuration'])])

        if 'tr' in train_config and train_config['tr'] != 'nnUNetTrainer':
            cmd.extend(['-tr', str(train_config['tr'])])

        if 'p' in train_config and train_config['p'] != 'nnUNetPlans':
            cmd.extend(['-p', str(train_config['p'])])

    # Add fold arguments to the command
    cmd.extend(['-f'] + [str(f) for f in folds])

    # Optional arguments with values
    if 'step_size' in config and str(config['step_size']) != '0.5':
        cmd.extend(['-step_size', str(config['step_size'])])

    if 'checkpoint_name' in config and config['checkpoint_name'] != 'checkpoint_final.pth':
        cmd.extend(['-chk', str(config['checkpoint_name'])])

    if 'num_processes_preprocessing' in config and str(config['num_processes_preprocessing']) != '3':
        cmd.extend(['-npp', str(config['num_processes_preprocessing'])])

    if 'num_processes_segmentation_export' in config and str(config['num_processes_segmentation_export']) != '3':
        cmd.extend(['-nps', str(config['num_processes_segmentation_export'])])

    if 'device' in config and str(config['device']) != 'cuda':
        cmd.extend(['-device', str(config['device'])])

    if 'prev_stage_predictions' in config and config['prev_stage_predictions'] is not None:
        cmd.extend(['-prev_stage_predictions', str(config['prev_stage_predictions'])])

    if 'num_parts' in config and str(config['num_parts']) != '1':
        cmd.extend(['-num_parts', str(config['num_parts'])])

    if 'part_id' in config and str(config['part_id']) != '0':
        cmd.extend(['-part_id', str(config['part_id'])])

    # Boolean flags
    if config.get('disable_tta', False):
        cmd.append('--disable_tta')

    if config.get('verbose', False):
        cmd.append('--verbose')

    if config.get('save_probabilities', False):
        cmd.append('--save_probabilities')

    if config.get('continue_prediction', False):
        cmd.append('--continue_prediction')

    if config.get('disable_progress_bar', False):
        cmd.append('--disable_progress_bar')

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
        print("Usage: python3 run_prediction.py config.yaml")
        sys.exit(1)

    yaml_file = sys.argv[1]
    config = read_config(yaml_file)

    # Get prediction configuration
    if 'predict' not in config:
        print("Error: 'predict' section not found in the config file")
        sys.exit(1)

    predict_config = config['predict']
    train_config = config['train']

    # Check for required parameters
    if 'input_folder' not in predict_config:
        print("Error: 'input_folder' is required in the predict config")
        sys.exit(1)

    if 'output_folder' not in predict_config:
        print("Error: 'output_folder' is required in the predict config")
        sys.exit(1)

    # Either model_folder or (dataset_name_or_id and configuration) must be present
    if 'model_folder' not in predict_config:
        if 'dataset_name_or_id' not in train_config:
            print("Error: Either 'model_folder' or 'dataset_name_or_id' must be specified")
            sys.exit(1)
        if 'configuration' not in train_config:
            print("Error: 'configuration' is required when using dataset_name_or_id")
            sys.exit(1)

    # Make sure output directory exists
    os.makedirs(predict_config['output_folder'], exist_ok=True)

    # Build and execute command
    cmd = build_predict_command(predict_config, train_config)
    execute_command(cmd)

    print(f"Prediction completed. Results saved to {predict_config['output_folder']}")


if __name__ == "__main__":
    main()