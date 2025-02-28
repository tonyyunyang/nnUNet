#!/usr/bin/env python3
import yaml
import subprocess
import sys


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


def build_command(config, fold):
    """Build the command to execute based on configuration with specific fold."""
    # Required arguments
    cmd = ['nnUNetv2_train']
    cmd.append(str(config['dataset_name_or_id']))
    cmd.append(str(config['configuration']))
    cmd.append(str(fold))  # Add fold in the correct position

    # Optional arguments with values
    if 'tr' in config and str(config['tr']) != 'nnUNetTrainer':
        cmd.extend(['-tr', str(config['tr'])])

    if 'p' in config and str(config['p']) != 'nnUNetPlans':
        cmd.extend(['-p', str(config['p'])])

    if 'pretrained_weights' in config and config['pretrained_weights'] is not None:
        cmd.extend(['-pretrained_weights', str(config['pretrained_weights'])])

    if 'num_gpus' in config and str(config['num_gpus']) != '1':
        cmd.extend(['-num_gpus', str(config['num_gpus'])])

    if 'device' in config and str(config['device']) != 'cuda':
        cmd.extend(['-device', str(config['device'])])

    # Boolean flags (no values)
    if config.get('npz', False):
        cmd.append('--npz')

    if config.get('c', False):
        cmd.append('--c')

    if config.get('val', False):
        cmd.append('--val')

    if config.get('val_best', False):
        cmd.append('--val_best')

    if config.get('disable_checkpointing', False):
        cmd.append('--disable_checkpointing')

    return cmd


def execute_command(cmd):
    """Execute the command and stream the output to the console."""
    cmd_str = ' '.join(cmd)
    print(f"Executing command: {cmd_str}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 run_training.py config.yaml")
        sys.exit(1)

    yaml_file = sys.argv[1]
    config = read_config(yaml_file)
    config = config['train']

    # Check for required parameters
    if 'dataset_name_or_id' not in config:
        print("Error: 'dataset_name_or_id' is required in the config file")
        sys.exit(1)
    if 'configuration' not in config:
        print("Error: 'configuration' is required in the config file")
        sys.exit(1)

    # Handle fold as array
    folds = config.get('fold', [0])
    if not isinstance(folds, list):
        folds = [folds]

    # Validate fold values (should be integers between 0 and 4)
    for fold in folds:
        try:
            fold_int = int(fold)
            if fold_int < 0 or fold_int > 4:
                print(f"Warning: Fold {fold} is out of the expected range (0-4)")
        except ValueError:
            print(f"Error: Fold {fold} is not a valid integer")
            sys.exit(1)

    for fold in folds:
        cmd = build_command(config, fold)
        execute_command(cmd)


if __name__ == "__main__":
    main()