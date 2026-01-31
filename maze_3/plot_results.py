import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob

# Configuration
RESULTS_DIR = "results/models"
OUTPUT_DIR = "results/plots"
SMOOTHING_WINDOW = 50  # Rolling average for smooth curves


def load_all_metrics(base_dir):
    """Walks through results dir and loads all metrics.json files."""
    experiments = []

    # Find all metrics.json files recursively
    files = glob(os.path.join(base_dir, "**", "metrics.json"), recursive=True)

    print(f"Found {len(files)} experiment files.")

    for filepath in files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Parse folder name to get metadata (E1_simple_PPO_...)
            folder_name = os.path.basename(os.path.dirname(filepath))
            parts = folder_name.split("_")

            # Robust parsing (assuming format: Env_Reward_Algo_Obs_Timestamp)
            env_id = parts[0]
            algo = parts[2]

            # Convert lists to Series for easy processing
            rewards = pd.Series(data["episode_rewards"])
            successes = pd.Series(data["episode_successes"])
            lengths = pd.Series(data["episode_lengths"])

            # Calculate cumulative steps (X-axis)
            steps = lengths.cumsum()

            experiments.append(
                {
                    "id": folder_name,
                    "env": env_id,
                    "algo": algo,
                    "steps": steps,
                    "rewards": rewards,
                    "success_rate": successes.rolling(
                        window=SMOOTHING_WINDOW, min_periods=1
                    ).mean(),
                    "lengths": lengths,
                    "final_success": successes.tail(100).mean(),  # Last 100 episodes
                }
            )
            print(f"Loaded: {folder_name}")

        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    return experiments


def plot_learning_curves(experiments):
    """Generates Success Rate vs Steps plot for each Environment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group by Environment
    env_groups = {}
    for exp in experiments:
        if exp["env"] not in env_groups:
            env_groups[exp["env"]] = []
        env_groups[exp["env"]].append(exp)

    # Create one plot per Environment
    for env_id, exps in env_groups.items():
        plt.figure(figsize=(10, 6))

        for exp in exps:
            # Downsample for faster plotting if needed
            plt.plot(exp["steps"], exp["success_rate"], label=f"{exp['algo']}")

        plt.title(f"Success Rate over Training ({env_id})")
        plt.xlabel("Total Timesteps")
        plt.ylabel(f"Success Rate (Rolling {SMOOTHING_WINDOW} eps)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.05, 1.05)

        save_path = os.path.join(OUTPUT_DIR, f"learning_curve_{env_id}.png")
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close()


def plot_final_comparison(experiments):
    """Bar chart comparing final performance of Algos on each Env."""
    data = []
    for exp in experiments:
        data.append(
            {
                "Environment": exp["env"],
                "Algorithm": exp["algo"],
                "Final Success Rate": exp["final_success"],
            }
        )

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Environment", y="Final Success Rate", hue="Algorithm")

    plt.title("Final Agent Performance Comparison")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "final_comparison_bar.png")
    plt.savefig(save_path)
    print(f"Saved comparison: {save_path}")
    plt.close()


def main():
    # 1. Load Data
    data = load_all_metrics(RESULTS_DIR)

    if not data:
        print("No data found! Check your results directory.")
        return

    # 2. Plot Learning Curves (Line Charts)
    plot_learning_curves(data)

    # 3. Plot Final Comparison (Bar Chart)
    plot_final_comparison(data)

    print("\nDone! Check the 'results/plots' folder.")


if __name__ == "__main__":
    main()
