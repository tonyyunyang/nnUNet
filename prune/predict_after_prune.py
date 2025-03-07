from pathlib import Path
import os
import torch
from torch.nn.utils import prune


def predict_after_prune(predictor, prune_config):
    predict_config = prune_config['predict']
    output_folder = predict_config['output_folder']

    # Save the model with pruning parameterization intact
    pruned_state_dict = predictor.network.state_dict()
    pruned_model_path = os.path.join(output_folder, "pruned_model_with_masks.pth")
    torch.save(pruned_state_dict, pruned_model_path)
    print(f"Saved pruned model with masks to: {pruned_model_path}")

    # Remove the pruning parameterization for prediction
    for name, module in predictor.network.named_modules():
        # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
        if hasattr(module, 'weight_orig'):
            print(f"Removing pruning parameterization for {module}")
            # Remove the reparameterization while keeping pruned weights at zero
            prune.remove(module, 'weight')
        if hasattr(module, 'bias_orig'):
            prune.remove(module, 'bias')
            print(f"Removing pruning parameterization for {module}")

    # Save the model with standard parameter structure
    standard_model_path = os.path.join(output_folder, "pruned_model_standard.pth")
    torch.save(predictor.network.state_dict(), standard_model_path)
    print(f"Saved pruned model with standard structure to: {standard_model_path}")

    # Get the pruned network state_dict
    pruned_state_dict = predictor.network.state_dict()

    # Update the list_of_parameters with the pruned weights
    fold = int(prune_config['fold'])
    if fold < len(predictor.list_of_parameters):
        # Update just the specific fold's parameters
        predictor.list_of_parameters[fold] = pruned_state_dict
    else:
        # If fold index doesn't match or we're using a single fold approach
        predictor.list_of_parameters = [pruned_state_dict]

    print(f"Updated predictor.list_of_parameters with pruned weights for fold {fold}")

    # Perform prediction with the modified model
    predictor.predict_from_files(
        predict_config['input_folder'],
        output_folder,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    return