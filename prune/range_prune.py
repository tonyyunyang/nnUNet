import torch
from torch.nn.utils import prune


class RangePruningMethod(prune.BasePruningMethod):
    """Prune weights that fall within a specified range of values.
    """
    PRUNING_TYPE = 'unstructured'  # Acts on individual weights

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def compute_mask(self, t, default_mask):
        """Compute a mask that zeros weights in the specified range"""
        mask = default_mask.clone()
        # Create mask where True means "prune this weight"
        range_mask = ((t >= self.min_val) & (t <= self.max_val))
        # Apply the mask (setting pruned weights to 0 in the mask)
        mask[range_mask] = 0
        return mask


def prune_weights_in_range(module, name, min_val, max_val):
    """Prunes weights in the specified parameter that fall within [min_val, max_val]"""
    RangePruningMethod.apply(module, name, min_val, max_val)
    return module


def apply_range_pruning_to_model(model, min_val: float, max_val: float):
    """Apply range-based pruning to all convolutional layers in a model."""
    pruned_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            if hasattr(module, 'weight'):
                # Count weights before pruning
                original_weight = module.weight.data.clone()
                total_layer = original_weight.numel()
                total_count += total_layer

                # Apply pruning
                prune_weights_in_range(module, 'weight', min_val, max_val)

                # Count pruned weights
                if hasattr(module, 'weight_mask'):
                    pruned_layer = (module.weight_mask == 0).sum().item()
                    pruned_count += pruned_layer
                    print(f"{name}: pruned {pruned_layer}/{total_layer} weights ({pruned_layer / total_layer:.2%})")

    # Print overall statistics
    if total_count > 0:
        print(f"Total pruned: {pruned_count}/{total_count} weights ({pruned_count / total_count:.2%})")

    return model