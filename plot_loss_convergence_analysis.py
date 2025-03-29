#!/usr/bin/env python3

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_losses_with_convergence(directory, fractions):
    """
    Searches for 'losses_*.csv' in `directory`, plots best_loss vs. iteration,
    draws lines connecting earliest crossing points for specified fractions,
    and fits/plots a linear regression for those points per fraction.
    """

    # 1. Find all CSV files matching 'losses_*.csv'
    pattern = os.path.join(directory, 'losses_*.csv')
    datafiles = sorted(glob.glob(pattern))

    if not datafiles:
        print(f"[INFO] No files found matching 'losses_*.csv' in {directory}")
        return

    print(f"[INFO] Found {len(datafiles)} file(s) in {directory}:")
    for f in datafiles:
        print(f"   - {os.path.basename(f)}")

    # We'll store each fraction's crossing points across runs:
    # fraction_points[fraction] = list of (iteration, best_loss)
    fraction_points = {f: [] for f in fractions}

    # 2. Start plotting
    plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')  # or another built-in style if you prefer

    # Plot each run
    for filepath in datafiles:
        # Read CSV
        df = pd.read_csv(filepath)

        # Plot the main curve for the run
        run_label = os.path.splitext(os.path.basename(filepath))[0]
        plt.plot(
            df['iteration'],
            df['best_loss'],
            marker='o',
            linestyle='-',
            label=run_label,
            alpha=0.85
        )

        # Compute min/max for best_loss
        max_val = df['best_loss'].max()
        min_val = df['best_loss'].min()
        diff = max_val - min_val

        # If diff <= 0, there's no "improvement" or data is invalid
        if diff <= 0:
            continue

        # 3. For each fraction, find earliest iteration that meets the threshold
        for p in fractions:
            threshold = max_val - p * diff
            # Find earliest index where best_loss <= threshold
            idx_candidates = df.index[df['best_loss'] <= threshold]
            if len(idx_candidates) > 0:
                earliest_idx = idx_candidates[0]
                x_val = df.loc[earliest_idx, 'iteration']
                y_val = df.loc[earliest_idx, 'best_loss']
                # Store it for this fraction
                fraction_points[p].append((x_val, y_val))

    # 5. Now do a linear regression for each fraction’s points and plot that line
    #    “iteration” is the predictor (X), “best_loss” is the response (y).
    for p in fractions:
        pts = fraction_points[p]
        if len(pts) < 2:
            # Need at least 2 points to fit a line
            continue

        xs = np.array([pt[0] for pt in pts], dtype=float)
        ys = np.array([pt[1] for pt in pts], dtype=float)

        # Fit a linear regression using numpy polyfit (degree=1 => a line)
        # Returns [slope, intercept]
        slope, intercept = np.polyfit(xs, ys, 1)

        # Let's create a smooth array of iteration values
        x_min, x_max = xs.min(), xs.max()
        x_range = np.linspace(x_min, x_max, 100)
        y_pred = slope * x_range + intercept

        # Plot that best-fit line
        # We'll use a dotted line style for clarity
        plt.plot(
            x_range,
            y_pred,
            linestyle=':',
            linewidth=2.5,
            label=f"LR p={p:.2f} (slope={slope:.4f})"
        )

    # 6. Formatting
    plt.title('Best Loss Over Iterations + Convergence + LR Lines', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Best Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    # 7. Save the figure in the same directory
    output_path = os.path.join(directory, 'best_loss_convergence_with_regression.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\n[INFO] Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search for losses_*.csv in a directory, plot best_loss vs iteration, "
            "annotate earliest crossing points for specified fractions, and "
            "perform linear regression on those points across runs."
        )
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing losses_*.csv files'
    )
    parser.add_argument(
        '--fractions',
        nargs='*',
        type=float,
        default=[0.3, 0.5, 0.8, 0.9, 0.95],
        help=(
            "List of fractions (floats) to determine earliest convergence lines. "
            "Default: 0.3 0.5 0.8 0.9 0.95"
        )
    )

    args = parser.parse_args()
    plot_losses_with_convergence(args.directory, args.fractions)

if __name__ == '__main__':
    main()
