from pathlib import Path
from helper.read_config import read_config, get_prune_config
from helper.load_model import choose_device, load_predictor_from_folder
from prune.range_prune import apply_range_pruning_to_model
from prune.verify_prune import verify_pruning
from pprint import pprint
from prune.predict_after_prune import predict_after_prune


def main():
    config_path = Path("scripts/config/config.yaml")
    device = choose_device()

    # Get the network architecture
    config = read_config(config_path)
    prune_config = get_prune_config(config)
    pprint(prune_config)
    predictor = load_predictor_from_folder(prune_config['model_folder'], prune_config['fold_tuple'], prune_config['checkpoint_name'])
    model = predictor.network

    # Inspect state_dict BEFORE pruning
    print("===== STATE DICT BEFORE PRUNING =====")
    before_keys = set()
    for key in model.state_dict().keys():
        before_keys.add(key)
        # Print a sample of keys to avoid overwhelming output
        if len(before_keys) <= 10 or 'encoder.stages.0.0.convs.0' in key:
            print(f"KEY: {key}")

    # print(list(model.named_parameters()))
    # print(model)
    verify_pruning(predictor.network)

    prune_model = apply_range_pruning_to_model(model, min_val=-5.0e-0, max_val=5.0e-0)

    # Inspect state_dict AFTER pruning
    print("\n===== STATE DICT AFTER PRUNING =====")
    after_keys = set()
    for key in prune_model.state_dict().keys():
        after_keys.add(key)
        # Print a sample of keys to avoid overwhelming output
        if len(after_keys) <= 10 or 'encoder.stages.0.0.convs.0' in key:
            print(f"KEY: {key}")

    # Show what's changed
    print("\n===== CHANGES IN STATE DICT =====")
    removed_keys = before_keys - after_keys
    added_keys = after_keys - before_keys

    print("REMOVED KEYS:")
    for key in removed_keys:
        if 'encoder.stages.0.0.convs.0' in key:
            print(f"- {key}")

    print("\nADDED KEYS:")
    for key in added_keys:
        if 'encoder.stages.0.0.convs.0' in key:
            print(f"+ {key}")

    _ = verify_pruning(predictor.network)

    predict_after_prune(predictor, prune_config)


if __name__ == "__main__":
    main()