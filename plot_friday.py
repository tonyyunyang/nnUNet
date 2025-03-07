import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_performance_vs_pruning(csv_file, output_dir="plots"):
    """
    Generate plots showing performance_Dice vs total pruning percentage for each fold,
    with separate lines for prune_bias True and False.

    Args:
        csv_file: Path to the CSV file containing the analysis results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    print(f"Reading data from {csv_file}")
    df = pd.read_csv(csv_file)

    # Print the first few rows to debug
    print("First few rows of the data:")
    print(df.head())

    # Print column names to confirm they exist
    print("Columns in the dataframe:", df.columns.tolist())

    # Check for missing data
    print("Missing values in key columns:")
    print(df[['fold', 'prune_method', 'prune_bias', 'performance_Dice', 'total_percentage']].isna().sum())

    # Convert total_percentage to numeric, handling errors
    df['total_percentage'] = pd.to_numeric(df['total_percentage'], errors='coerce')

    # Check the range of total_percentage to determine if it needs scaling
    max_percentage = df['total_percentage'].max()
    print(f"Maximum total_percentage value: {max_percentage}")

    # If the percentage is already in 0-100 range, divide by 100
    if max_percentage > 1.0:
        print("Scaling total_percentage from 0-100 to 0-1 range")
        df['total_percentage'] = df['total_percentage'] / 100.0

    # Convert performance_Dice to numeric
    df['performance_Dice'] = pd.to_numeric(df['performance_Dice'], errors='coerce')

    # Get unique folds
    folds = df['fold'].unique()
    print(f"Found {len(folds)} unique folds: {sorted(folds)}")

    # Define colors and bias labels directly where used
    bias_colors = {True: 'blue', False: 'red'}
    bias_labels = {True: 'With Bias Pruning', False: 'Without Bias Pruning'}

    # Create a plot for each fold
    for fold in sorted(folds):
        print(f"Processing fold {fold}")

        # Filter data for this fold
        fold_data = df[df['fold'] == fold].copy()
        print(f"Found {len(fold_data)} data points for fold {fold}")

        if len(fold_data) == 0:
            print(f"No data for fold {fold}, skipping")
            continue

        # Create a new figure
        plt.figure(figsize=(10, 8))

        # Get unique prune_bias values
        bias_values = fold_data['prune_bias'].dropna().unique()
        print(f"Unique bias values for fold {fold}: {bias_values}")

        # For each unique prune method, use a different marker
        prune_methods = fold_data['prune_method'].unique()
        markers = {method: marker for method, marker in
                   zip(prune_methods, ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*'])}

        # For each bias setting, plot a line
        legend_handles = []  # To store handles for legend
        legend_labels = []  # To store labels for legend

        for bias in bias_values:
            # Get data for this bias setting
            bias_data = fold_data[fold_data['prune_bias'] == bias]
            print(f"  - Found {len(bias_data)} data points for bias={bias}")

            if len(bias_data) == 0:
                continue

            # Group by prune_method to use different markers
            for method in bias_data['prune_method'].unique():
                method_data = bias_data[bias_data['prune_method'] == method]

                # Sort by total_percentage for proper line drawing
                method_data = method_data.sort_values(by='total_percentage')

                # Check if we have enough data to plot
                if len(method_data) == 0:
                    continue

                print(f"    - Plotting {len(method_data)} points for method={method}, bias={bias}")
                print(
                    f"      Data range: x={method_data['total_percentage'].min()}-{method_data['total_percentage'].max()}, y={method_data['performance_Dice'].min()}-{method_data['performance_Dice'].max()}")

                # Get the color for this bias setting
                color = bias_colors.get(bias, 'green')

                # Create a label for the legend
                label = f"{method} - {'With' if bias else 'Without'} Bias"

                # Plot the line
                line, = plt.plot(
                    method_data['total_percentage'],
                    method_data['performance_Dice'],
                    marker=markers.get(method, 'o'),
                    linestyle='-' if bias else '--',
                    color=color,
                    linewidth=2,
                    markersize=8,
                    label=label
                )

                # Store handle and label for legend
                legend_handles.append(line)
                legend_labels.append(label)

                # Add data point annotations, but only for without bias (False) points
                # or only once per min_val/max_val pair
                if not bias:  # Only annotate points without bias pruning
                    for i, row in method_data.iterrows():
                        min_val = row['min_val']
                        max_val = row['max_val']
                        # Format min_val and max_val to be shorter
                        if isinstance(min_val, str) and 'e' in min_val:
                            min_val = min_val.replace('e-', 'e-')
                        if isinstance(max_val, str) and 'e' in max_val:
                            max_val = max_val.replace('e-', 'e-')

                        label = f"{min_val},{max_val}"
                        plt.annotate(
                            label,
                            (row['total_percentage'], row['performance_Dice']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=8
                        )

        # Set plot title and labels
        plt.title(f'Fold {fold}: Dice Performance vs Pruning Percentage', fontsize=16)
        plt.xlabel('Total Pruning Percentage', fontsize=14)
        plt.ylabel('Dice Performance', fontsize=14)

        # Set the axes limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        # Set tick marks at 0.1 intervals
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend if we have handles
        if legend_handles:
            plt.legend(legend_handles, legend_labels, loc='lower left')
        else:
            print(f"Warning: No legend entries for fold {fold}")

        # Save the plot
        output_file = os.path.join(output_dir, f'fold_{fold}_performance_vs_pruning.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot for fold {fold} to {output_file}")

        # Close the figure to free memory
        plt.close()

    print(f"All plots saved to {output_dir}")


def create_summary_plot(csv_file, output_dir="plots"):
    """
    Create a summary plot showing the average performance across all folds

    Args:
        csv_file: Path to the CSV file containing the analysis results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert string values to appropriate types
    df['total_percentage'] = pd.to_numeric(df['total_percentage'], errors='coerce')
    if df['total_percentage'].max() > 1.0:
        df['total_percentage'] = df['total_percentage'] / 100.0

    df['performance_Dice'] = pd.to_numeric(df['performance_Dice'], errors='coerce')

    # Group by prune_method, min_val, max_val, and prune_bias to get averages
    grouped = df.groupby(['prune_method', 'min_val', 'max_val', 'prune_bias']).agg({
        'total_percentage': 'mean',
        'performance_Dice': 'mean'
    }).reset_index()

    # Create a new figure
    plt.figure(figsize=(12, 10))

    # Colors for different pruning methods
    method_colors = {
        'RangePruning': 'blue',
        'MagnitudePruning': 'red',
        'RandomPruning': 'green',
        # Add more methods if needed
    }

    # Markers for different bias settings
    bias_markers = {True: 'o', False: 's'}

    # For each pruning method and bias setting, plot a line
    legend_handles = []
    legend_labels = []

    for method in grouped['prune_method'].unique():
        method_data = grouped[grouped['prune_method'] == method]

        for bias in sorted(method_data['prune_bias'].unique()):
            # Get data for this combination
            combo_data = method_data[method_data['prune_bias'] == bias]

            # Sort by total_percentage for proper line drawing
            combo_data = combo_data.sort_values(by='total_percentage')

            # Skip if no data
            if len(combo_data) == 0:
                continue

            # Create a label
            label = f"{method} - {'With' if bias else 'Without'} Bias Pruning"

            # Plot the line
            line, = plt.plot(
                combo_data['total_percentage'],
                combo_data['performance_Dice'],
                marker=bias_markers.get(bias, 'x'),
                linestyle='-' if bias else '--',
                color=method_colors.get(method, 'purple'),
                linewidth=2,
                markersize=8,
                label=label
            )

            # Add to legend
            legend_handles.append(line)
            legend_labels.append(label)

    # Set plot title and labels
    plt.title('Average Performance vs Pruning Percentage Across All Folds', fontsize=16)
    plt.xlabel('Total Pruning Percentage', fontsize=14)
    plt.ylabel('Dice Performance', fontsize=14)

    # Set the axes limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Set tick marks at 0.1 intervals
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend if we have handles
    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc='best')

    # Save the plot
    output_file = os.path.join(output_dir, 'average_performance_vs_pruning.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {output_file}")

    # Close the figure
    plt.close()


if __name__ == "__main__":
    csv_file = "pruning_analysis_results.csv"
    output_dir = "pruning_plots"

    # Create individual fold plots
    plot_performance_vs_pruning(csv_file, output_dir)

    # Create summary plot
    create_summary_plot(csv_file, output_dir)

    print("Visualization complete!")