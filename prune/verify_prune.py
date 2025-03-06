import torch


def verify_pruning(model):
    """
    Verify that pruning was successfully applied to the model.
    Returns statistics about pruning.
    """
    pruned_layers = 0
    total_zeros = 0
    total_weights = 0

    # Check each module for pruning masks
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            if hasattr(module, 'weight_mask'):
                pruned_layers += 1

                # Count zeros in the effective weights (original weights * mask)
                effective_weights = module.weight_orig * module.weight_mask
                zeros = (effective_weights == 0).sum().item()
                total = effective_weights.numel()

                total_zeros += zeros
                total_weights += total

                # Print per-layer statistics
                sparsity = 100.0 * zeros / total
                print(f"Layer {name}: {zeros}/{total} weights pruned ({sparsity:.2f}% sparsity)")

    # Print overall statistics
    if total_weights > 0:
        overall_sparsity = 100.0 * total_zeros / total_weights
        print(f"\nOverall pruning: {total_zeros}/{total_weights} weights pruned "
              f"({overall_sparsity:.2f}% sparsity)")
        print(f"Number of pruned layers: {pruned_layers}")

        if pruned_layers == 0:
            print("WARNING: No pruned layers found! Pruning may not have been applied.")
            return False
        return True
    else:
        print("No convolutional layers / weight masks found in the model.")
        return False