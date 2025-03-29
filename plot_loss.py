import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses_from_directory(directory):
    # Find all CSV files starting with 'losses_' and ending with '.csv'
    pattern = os.path.join(directory, 'losses_*.csv')
    datafiles = sorted(glob.glob(pattern))

    if not datafiles:
        print(f"No files found matching 'losses_*.csv' in {directory}")
        return

    print(f"Found {len(datafiles)} files:")
    for f in datafiles:
        print(f" - {os.path.basename(f)}")

    # --- 1) Plot with a linear x-axis ---
    plt.figure(figsize=(12, 7))

    for filepath in datafiles:
        df = pd.read_csv(filepath)
        label = os.path.splitext(os.path.basename(filepath))[0]

        # Plot the main line
        line_obj, = plt.plot(
            df['iteration'],
            df['best_loss'],
            marker=None,
            linestyle='-',
            label=label
        )

        # Calculate the average best_loss
        avg_loss = df['best_loss'].mean()

        # Add a dashed horizontal line at the average, in the same color
        plt.axhline(
            y=avg_loss,
            color=line_obj.get_color(),  # same color as the main line
            linestyle='--',
            alpha=0.7,
            label=f"{label} avg: {avg_loss:.3f}"
        )

    # Formatting (linear)
    plt.title('Best Loss (Linear Scale)', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Best Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Run', fontsize=12, title_fontsize=13)
    plt.tight_layout()

    # Save linear plot
    output_path_linear = os.path.join(directory, 'best_loss_comparison.png')
    plt.savefig(output_path_linear, dpi=300)
    plt.close()

    print(f"\n✅ Linear-scale plot saved to: {output_path_linear}")

    # --- 2) Plot with a log-scale x-axis ---
    plt.figure(figsize=(12, 7))

    for filepath in datafiles:
        df = pd.read_csv(filepath)
        label = os.path.splitext(os.path.basename(filepath))[0]

        # Plot the main line
        line_obj, = plt.plot(
            df['iteration'],
            df['best_loss'],
            marker=None,
            linestyle='-',
            label=label
        )

        # Calculate the average best_loss
        avg_loss = df['best_loss'].mean()

        # Add a dashed horizontal line at the average, in the same color
        plt.axhline(
            y=avg_loss,
            color=line_obj.get_color(),
            linestyle='--',
            alpha=0.7,
            label=f"{label} avg: {avg_loss:.3f}"
        )

    # Apply log scale on the x-axis
    plt.xscale('log')

    # Formatting (log)
    plt.title('Best Loss (Log Scale)', fontsize=16)
    plt.xlabel('Iteration (log scale)', fontsize=14)
    plt.ylabel('Best Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Run', fontsize=12, title_fontsize=13)
    plt.tight_layout()

    # Save log-scale plot
    output_path_log = os.path.join(directory, 'best_loss_comparison_logscale.png')
    plt.savefig(output_path_log, dpi=300)
    plt.close()

    print(f"✅ Log-scale plot saved to: {output_path_log}")

def main():
    parser = argparse.ArgumentParser(description='Plot best_loss vs iteration from multiple CSV files, including an average line.')
    parser.add_argument('directory', type=str, help='Directory containing losses_*.csv files')

    args = parser.parse_args()
    plot_losses_from_directory(args.directory)

if __name__ == '__main__':
    main()
