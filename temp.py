import os
import json
import re
from pathlib import Path


def process_directories(root_dir):
    """Process all directories and extract required information."""
    results = []

    # Walk through all folds
    for fold_dir in Path(root_dir).glob('fold_*'):
        fold_num = int(fold_dir.name.split('_')[1])
        model_dir = fold_dir / 'final_model'

        if not model_dir.exists():
            # Try best_model if final_model doesn't exist
            model_dir = fold_dir / 'best_model'
            if not model_dir.exists():
                print(f"No final_model or best_model directory found in {fold_dir}")
                continue

        # Process each pruning directory
        for pruning_dir in model_dir.iterdir():
            if not pruning_dir.is_dir():
                continue

            # Parse directory name
            parts = pruning_dir.name.split('__')
            if len(parts) < 3:
                print(f"Skipping directory with unexpected format: {pruning_dir.name}")
                continue

            prune_method = parts[0]
            min_val = parts[1]
            max_val = parts[2]

            prune_bias = None
            if len(parts) > 3:
                if parts[3] == 'bias':
                    prune_bias = True
                elif parts[3] == 'no_bias':
                    prune_bias = False

            # Get performance from summary.json
            performance = {}
            summary_path = pruning_dir / 'summary.json'
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                        performance = data.get('foreground_mean', {})
                    print(f"Extracted performance metrics from {summary_path}")
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
            else:
                print(f"Warning: {summary_path} does not exist")

            # Get pruning stats from zero_parameter_analysis.txt
            pruning_stats = {}
            zero_path = pruning_dir / 'zero_parameter_analysis.txt'
            if zero_path.exists():
                try:
                    with open(zero_path, 'r') as f:
                        lines = f.readlines()

                        # Find the summary section
                        summary_index = -1
                        for i, line in enumerate(lines):
                            if line.strip() == "SUMMARY:":
                                summary_index = i
                                break

                        if summary_index != -1 and summary_index + 3 < len(lines):
                            weights_line = lines[summary_index + 1].strip()
                            biases_line = lines[summary_index + 2].strip()
                            total_line = lines[summary_index + 3].strip()

                            # Extract weights percentage
                            weights_match = re.search(r"Weights:.*\(([\d.]+)%\)", weights_line)
                            if weights_match:
                                pruning_stats['weights_percentage'] = float(weights_match.group(1))

                                # Also extract the fraction
                                fraction_match = re.search(r"Weights:\s+([\d,]+)/([\d,]+)", weights_line)
                                if fraction_match:
                                    zeros = int(fraction_match.group(1).replace(',', ''))
                                    total = int(fraction_match.group(2).replace(',', ''))
                                    pruning_stats['weights_zeros'] = zeros
                                    pruning_stats['weights_total'] = total

                            # Extract biases percentage
                            biases_match = re.search(r"Biases:.*\(([\d.]+)%\)", biases_line)
                            if biases_match:
                                pruning_stats['biases_percentage'] = float(biases_match.group(1))

                                # Also extract the fraction
                                fraction_match = re.search(r"Biases:\s+([\d,]+)/([\d,]+)", biases_line)
                                if fraction_match:
                                    zeros = int(fraction_match.group(1).replace(',', ''))
                                    total = int(fraction_match.group(2).replace(',', ''))
                                    pruning_stats['biases_zeros'] = zeros
                                    pruning_stats['biases_total'] = total

                            # Extract total percentage
                            total_match = re.search(r"Total:.*\(([\d.]+)%\)", total_line)
                            if total_match:
                                pruning_stats['total_percentage'] = float(total_match.group(1))

                                # Also extract the fraction
                                fraction_match = re.search(r"Total:\s+([\d,]+)/([\d,]+)", total_line)
                                if fraction_match:
                                    zeros = int(fraction_match.group(1).replace(',', ''))
                                    total = int(fraction_match.group(2).replace(',', ''))
                                    pruning_stats['total_zeros'] = zeros
                                    pruning_stats['total_total'] = total

                    print(f"Extracted pruning stats from {zero_path}")
                except Exception as e:
                    print(f"Error reading {zero_path}: {e}")
            else:
                print(f"Warning: {zero_path} does not exist")

            # Save results
            result = {
                'fold': fold_num,
                'prune_method': prune_method,
                'min_val': min_val,
                'max_val': max_val,
                'prune_bias': prune_bias,
                'performance': performance,
                'pruning_stats': pruning_stats
            }

            results.append(result)
            print(f"Processed {pruning_dir.name} in fold {fold_num}")

    return results


def save_results(results, json_file, csv_file):
    """Save results to JSON and CSV files."""
    # Save as JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_file}")

    # Save as CSV
    with open(csv_file, 'w') as f:
        # Write header
        header = ['fold', 'prune_method', 'min_val', 'max_val', 'prune_bias']

        # Get all performance keys
        performance_keys = set()
        for result in results:
            performance_keys.update(result.get('performance', {}).keys())

        for key in sorted(performance_keys):
            header.append(f'performance_{key}')

        # Add pruning stats
        header.extend([
            'weights_zeros', 'weights_total', 'weights_percentage',
            'biases_zeros', 'biases_total', 'biases_percentage',
            'total_zeros', 'total_total', 'total_percentage'
        ])

        f.write(','.join(header) + '\n')

        # Write data rows
        for result in results:
            row = [
                str(result['fold']),
                result['prune_method'],
                result['min_val'],
                result['max_val'],
                str(result['prune_bias'])
            ]

            # Add performance values
            for key in sorted(performance_keys):
                row.append(str(result.get('performance', {}).get(key, '')))

            # Add pruning stats
            pruning_stats = result.get('pruning_stats', {})
            row.extend([
                str(pruning_stats.get('weights_zeros', '')),
                str(pruning_stats.get('weights_total', '')),
                str(pruning_stats.get('weights_percentage', '')),
                str(pruning_stats.get('biases_zeros', '')),
                str(pruning_stats.get('biases_total', '')),
                str(pruning_stats.get('biases_percentage', '')),
                str(pruning_stats.get('total_zeros', '')),
                str(pruning_stats.get('total_total', '')),
                str(pruning_stats.get('total_percentage', ''))
            ])

            f.write(','.join(row) + '\n')

    print(f"Saved CSV results to {csv_file}")


if __name__ == "__main__":
    root_directory = "nnUNet_results_prune_test/Dataset027_ACDC/FlexibleTrainerV1__nnUNetPlans__2d/predictions"  # Change this to the actual path if needed
    json_file = "pruning_analysis_results.json"
    csv_file = "pruning_analysis_results.csv"

    print(f"Starting analysis of directory: {root_directory}")
    results = process_directories(root_directory)
    save_results(results, json_file, csv_file)

    print(f"Processed {len(results)} model directories")
    print(f"Results saved to {json_file} and {csv_file}")