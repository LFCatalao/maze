"""
Compare metrics across different experiment configurations.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(results_dir: str):
    """Load all metrics.json files from results directory."""
    experiments = {}

    for exp_folder in Path(results_dir).iterdir():
        if exp_folder.is_dir():
            metrics_file = exp_folder / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    experiments[exp_folder.name] = json.load(f)

    return experiments


def plot_learning_curves(experiments: dict, window: int = 50):
    """Plot smoothed success rate over episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Success rate
    ax = axes[0, 0]
    for name, data in experiments.items():
        successes = data["episode_successes"]
        smoothed = np.convolve(successes, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=name.split("_")[0])  # E1, E2, etc.
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Success Rate (smoothed, window={window})")
    ax.legend()
    ax.grid(True)

    # Episode length
    ax = axes[0, 1]
    for name, data in experiments.items():
        lengths = data["episode_lengths"]
        smoothed = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=name.split("_")[0])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.legend()
    ax.grid(True)

    # Key pickup rate
    ax = axes[1, 0]
    for name, data in experiments.items():
        pickups = data["key_pickups"]
        if sum(pickups) > 0:  # Only plot if keys exist
            smoothed = np.convolve(pickups, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, label=name.split("_")[0])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Pickup Rate")
    ax.set_title("Key Pickup Rate")
    ax.legend()
    ax.grid(True)

    # Door open rate
    ax = axes[1, 1]
    for name, data in experiments.items():
        opens = data["door_opens"]
        if sum(opens) > 0:
            smoothed = np.convolve(opens, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, label=name.split("_")[0])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Open Rate")
    ax.set_title("Door Open Rate")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_curves.png", dpi=150)
    plt.show()


def print_summary_table(experiments: dict):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    header = f"{'Experiment':<25} {'Success%':>10} {'Reward':>10} {'Length':>10} {'Key%':>8} {'Door%':>8}"
    print(header)
    print("-" * 80)

    for name, data in sorted(experiments.items()):
        s = data["summary"]
        print(
            f"{name:<25} {s['final_100_success_rate']*100:>9.1f}% {s['mean_reward']:>10.1f} {s['mean_length']:>10.1f} {s['key_pickup_rate']*100:>7.1f}% {s['door_open_rate']*100:>7.1f}%"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="results/models", help="Directory with experiment results"
    )
    parser.add_argument("--window", type=int, default=50, help="Smoothing window size")
    args = parser.parse_args()

    experiments = load_metrics(args.results_dir)

    if not experiments:
        print(f"No metrics found in {args.results_dir}")
        return

    print(f"Found {len(experiments)} experiments")
    print_summary_table(experiments)
    plot_learning_curves(experiments, window=args.window)


if __name__ == "__main__":
    main()
