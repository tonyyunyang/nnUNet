import os
from pathlib import Path

import yaml

from helper.read_config import read_config, get_prune_config
from helper.load_model import choose_device, load_predictor_from_folder
from prune.range_prune import apply_range_pruning_to_model
from prune.verify_prune import verify_pruning
from pprint import pprint
from prune.predict_after_prune import predict_after_prune


def count_zero_parameters(model, output_path):
    """
    Count the number of zero-valued weights and biases in a model and their proportions.
    Save the results to a text file at output_path.

    Args:
        model: PyTorch model
        output_path: Path to save the output text file

    Returns:
        A tuple of (weight_stats, bias_stats, total_stats) each containing:
        (num_zeros, total_params, proportion_zeros)
    """
    output_path = os.path.join(output_path, "zero_parameter_analysis.txt")

    weight_zeros = 0
    weight_total = 0
    bias_zeros = 0
    bias_total = 0

    # Prepare the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a list to store all output lines (for both console and file)
    output_lines = []

    # Function to both print and store a line
    def log_line(line):
        print(line)
        output_lines.append(line + "\n")

    # Iterate through all modules that have parameters
    for name, module in model.named_modules():
        # Handle weights
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            # Count zeros in weights
            w_zeros = (weight == 0).sum().item()
            w_total = weight.numel()

            weight_zeros += w_zeros
            weight_total += w_total

            if w_zeros > 0:
                log_line(f"{name}.weight: {w_zeros}/{w_total} zeros ({w_zeros / w_total:.2%})")

        # Handle biases
        if hasattr(module, 'bias') and module.bias is not None:
            bias = module.bias.data
            # Count zeros in biases
            b_zeros = (bias == 0).sum().item()
            b_total = bias.numel()

            bias_zeros += b_zeros
            bias_total += b_total

            if b_zeros > 0:
                log_line(f"{name}.bias: {b_zeros}/{b_total} zeros ({b_zeros / b_total:.2%})")

    # Calculate totals
    total_zeros = weight_zeros + bias_zeros
    total_params = weight_total + bias_total
    weight_proportion = weight_zeros / weight_total if weight_total > 0 else 0
    bias_proportion = bias_zeros / bias_total if bias_total > 0 else 0
    total_proportion = total_zeros / total_params if total_params > 0 else 0

    # Print and log summary
    log_line("\nSUMMARY:")
    log_line(f"Weights: {weight_zeros:,}/{weight_total:,} zeros ({weight_proportion:.2%})")
    log_line(f"Biases:  {bias_zeros:,}/{bias_total:,} zeros ({bias_proportion:.2%})")
    log_line(f"Total:   {total_zeros:,}/{total_params:,} zeros ({total_proportion:.2%})")

    # Save all output to the file
    try:
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
        print(f"\nZero parameter analysis saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving zero parameter analysis to file: {e}")

    weight_stats = (weight_zeros, weight_total, weight_proportion)
    bias_stats = (bias_zeros, bias_total, bias_proportion)
    total_stats = (total_zeros, total_params, total_proportion)

    return weight_stats, bias_stats, total_stats


def format_scientific(num):
    # Convert to scientific notation with 0 decimal places
    formatted = f"{num:.0e}"
    # Remove leading zeros in the exponent part
    parts = formatted.split('e')
    exponent = parts[1].lstrip('+-0')
    # Rebuild with correct sign
    if parts[1].startswith('-'):
        return f"{parts[0]}e-{exponent}"
    else:
        return f"{parts[0]}e+{exponent}"


def update_output_folder_in_config(config_path, output_folder):
    """
    Updates only the output_folder field in the config file without changing anything else.

    Args:
        config_path: Path to the config.yaml file
        output_folder: The output folder path to set
    """
    try:
        # Load the current config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Ensure the structure exists
        if 'prune' not in config:
            config['prune'] = {}
        if 'predict' not in config['prune']:
            config['prune']['predict'] = {}

        # Update only the output_folder field
        config['prune']['predict']['output_folder'] = output_folder

        # Write the config back to file
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Successfully updated output_folder in config: {output_folder}")

    except Exception as e:
        print(f"Error updating output_folder in config: {e}")


def main():
    config_path = Path("scripts/config/config.yaml")

    # Get the network architecture
    config = read_config(config_path)
    prune_config = get_prune_config(config)
    pprint(prune_config)
    predictor = load_predictor_from_folder(prune_config['model_folder'], prune_config['fold_tuple'], prune_config['checkpoint_name'])

    # if prune_config['predict'].get('output_folder', None) is None:
    prune_config['predict']['output_folder'] = os.path.join(prune_config['model_folder'], 'predictions', f"fold_{prune_config['fold']}")
    if prune_config['checkpoint_name'] == 'checkpoint_best.pth':
        prune_config['predict']['output_folder'] = os.path.join(prune_config['predict']['output_folder'], 'best_model')
    else:
        prune_config['predict']['output_folder'] = os.path.join(prune_config['predict']['output_folder'], 'final_model')
    prune_config['predict']['output_folder'] = os.path.join(prune_config['predict']['output_folder'], f"{prune_config['prune_method']}__"
                                                                                f"{format_scientific(prune_config['prune_parameters']['min_val'])}__"
                                                                                f"{format_scientific(prune_config['prune_parameters']['max_val'])}__"
                                                                                f"{'bias' if prune_config['prune_parameters']['prune_bias'] else 'no_bias'}")
    os.makedirs(prune_config['predict']['output_folder'], exist_ok=True)


    verify_pruning(predictor.network)

    _ = apply_range_pruning_to_model(predictor.network, min_val=prune_config['prune_parameters']['min_val'],
                                     max_val=prune_config['prune_parameters']['max_val'],
                                     prune_bias=prune_config['prune_parameters']['prune_bias'])

    predict_after_prune(predictor, prune_config)

    print("\nDetailed zero parameter analysis:")
    weight_stats, bias_stats, total_stats = count_zero_parameters(predictor.network, prune_config['predict']['output_folder'])
    print(f"\nPruning resulted in {total_stats[2]:.2%} of all parameters set to zero")

    # Update the original config with the new output folder
    if 'prune' not in config:
        config['prune'] = {}
    if 'predict' not in config['prune']:
        config['prune']['predict'] = {}

    # Update the output_folder in the config
    config['prune']['predict']['output_folder'] = prune_config['predict']['output_folder']

    # Add this line to update only the output_folder in the config
    update_output_folder_in_config(config_path, prune_config['predict']['output_folder'])


if __name__ == "__main__":
    main()