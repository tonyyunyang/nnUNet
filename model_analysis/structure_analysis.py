import torch
from pathlib import Path


def inspect_pth_model(model_path):
    model_path = Path(model_path)

    # Check if file exists
    if not model_path.exists():
        print(f"Error: File {model_path} does not exist")
        return

    # Check if file has .pth extension
    if model_path.suffix != '.pth':
        print(f"Warning: File {model_path} does not have a .pth extension")

    print(f"\n{'=' * 60}")
    print(f"INSPECTING MODEL: {model_path}")
    print(f"{'=' * 60}\n")

    # Load the model
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print(f"Successfully loaded model from {model_path}\n")
    except Exception as e:
        print(f"Failed to load model from {model_path}\n"
              f"Error: {e}\n")
        return

    if isinstance(checkpoint, dict):
        print(f"Checkpoint is a dictionary with keys: {list(checkpoint.keys())}")
        for key, value in checkpoint.items():
            print(f"\n{'-' * 60}")
            print(f"Key: {key}")
            print(f"Value type: {type(value)}")
        if 'network_weights' in checkpoint:
            analyze_network_weights(checkpoint['network_weights'])
    else:
        print(f"Checkpoint is not a dictionary")
        return

    print(f"\n{'=' * 60}")
    print(f"END OF MODEL INSPECTION")
    print(f"{'=' * 60}")


def analyze_network_weights(network_weights):
    print(f"\n{'-' * 60}")
    print(f"NETWORK WEIGHTS ANALYSIS")
    print(f"{'-' * 60}")

    total_params = 0
    layer_params = {}

    print("\nLayer Parameters Summary:")
    print(f"{'Layer Name':<50} {'Shape':<25} {'Size':<15} {'Mean':<10} {'Std':<10}")
    print("-" * 110)

    for name, param in network_weights.items():
        # Discard all_modules, because it is redundant, it saves the same layer twice
        if "all_modules" not in name:
            # Count parameters
            param_size = param.numel()
            total_params += param_size

            # Store layer info
            layer_params[name] = {
                'shape': param.shape,
                'size': param_size,
                'mean': param.mean().item(),
                'std': param.std().item() if param_size > 1 else 0
            }
            print(
                f"{name:<50} {str(param.shape):<25} {param_size:<15} {param.mean().item():<10.4f} {param.std().item():<10.4f}")

    # Print summary
    print(f"\n{'-' * 60}")
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Layer types:")


# Usage
if __name__ == "__main__":
    path = ""
    model_path = "../nnUNet_results/Dataset027_ACDC/FlexibleTrainerV1__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
    inspect_pth_model(model_path)