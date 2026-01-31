"""
Generate publication-ready figures for report.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


# Style settings
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
    }
)

COLORS = {
    "PPO": "#2ecc71",
    "DQN": "#3498db",
    "A2C": "#e74c3c",
}


def load_all_experiments(results_dir: str) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    records = []

    for exp_folder in Path(results_dir).iterdir():
        if not exp_folder.is_dir():
            continue

        metrics_file = exp_folder / "metrics.json"
        config_file = exp_folder / "config.txt"

        if not metrics_file.exists():
            continue

        # Skip empty or corrupted files
        try:
            with open(metrics_file) as f:
                content = f.read().strip()
                if not content:
                    print(f"Warning: Empty file, skipping: {metrics_file}")
                    continue
                metrics = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Corrupted JSON, skipping: {metrics_file}")
            print(f"  Error: {e}")
            continue

        # Parse config
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    if ":" in line:
                        key, val = line.strip().split(":", 1)
                        config[key.strip().lower()] = val.strip()

        # Build record
        record = {
            "experiment": exp_folder.name,
            "env": config.get("environment", "unknown"),
            "algo": config.get("algorithm", "unknown"),
            "seed": int(config.get("seed", 0)),
            "final_100_success_rate": metrics.get("summary", {}).get("final_100_success_rate", 0),
            "mean_reward": metrics.get("summary", {}).get("mean_reward", 0),
            "mean_length": metrics.get("summary", {}).get("mean_length", 0),
            "key_pickup_rate": metrics.get("summary", {}).get("key_pickup_rate", 0),
            "door_open_rate": metrics.get("summary", {}).get("door_open_rate", 0),
            "episode_rewards": metrics.get("episode_rewards", []),
            "episode_successes": metrics.get("episode_successes", []),
            "episode_lengths": metrics.get("episode_lengths", []),
        }
        records.append(record)

    if not records:
        print(f"No valid experiments found in {results_dir}")
        return None

    return pd.DataFrame(records)


def figure_1_learning_curves(df: pd.DataFrame, output_dir: str):
    """
    Figure 1: Learning curves comparing algorithms across environments.
    Grid of subplots: rows=environments, cols=metrics
    """
    envs = sorted(df["env"].unique())
    n_envs = len(envs)

    fig, axes = plt.subplots(n_envs, 2, figsize=(12, 3 * max(n_envs, 1)))

    # Handle single environment case (axes is 1D)
    if n_envs == 1:
        axes = axes.reshape(1, -1)

    for i, env in enumerate(envs):
        env_data = df[df["env"] == env]

        # Success rate curves
        ax = axes[i, 0]
        for algo in ["PPO", "DQN", "A2C"]:
            algo_data = env_data[env_data["algo"] == algo]
            if algo_data.empty:
                continue

            # Average across seeds
            all_successes = [
                np.array(r, dtype=float) for r in algo_data["episode_successes"] if len(r) > 0
            ]
            if len(all_successes) == 0:
                continue

            # Pad to same length (convert to float first for NaN support)
            max_len = max(len(s) for s in all_successes)
            padded = np.array(
                [
                    np.pad(s.astype(float), (0, max_len - len(s)), constant_values=np.nan)
                    for s in all_successes
                ]
            )

            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)

            # Smooth
            window = min(50, max(1, len(mean) // 10))
            if window > 1 and len(mean) > window:
                mean_smooth = np.convolve(mean, np.ones(window) / window, mode="valid")
                std_smooth = np.convolve(std, np.ones(window) / window, mode="valid")
            else:
                mean_smooth = mean
                std_smooth = std

            ax.plot(mean_smooth, label=algo, color=COLORS[algo])
            ax.fill_between(
                range(len(mean_smooth)),
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                alpha=0.2,
                color=COLORS[algo],
            )

        ax.set_ylabel("Success Rate")
        ax.set_title(f"{env}: Success Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Episode length curves
        ax = axes[i, 1]
        for algo in ["PPO", "DQN", "A2C"]:
            algo_data = env_data[env_data["algo"] == algo]
            if algo_data.empty:
                continue

            all_lengths = [
                np.array(r, dtype=float) for r in algo_data["episode_lengths"] if len(r) > 0
            ]
            if len(all_lengths) == 0:
                continue

            max_len = max(len(l) for l in all_lengths)
            padded = np.array(
                [
                    np.pad(l.astype(float), (0, max_len - len(l)), constant_values=np.nan)
                    for l in all_lengths
                ]
            )

            mean = np.nanmean(padded, axis=0)
            window = min(50, max(1, len(mean) // 10))
            if window > 1 and len(mean) > window:
                mean_smooth = np.convolve(mean, np.ones(window) / window, mode="valid")
            else:
                mean_smooth = mean

            ax.plot(mean_smooth, label=algo, color=COLORS[algo])

        ax.set_ylabel("Episode Length")
        ax.set_title(f"{env}: Episode Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Episode")
    axes[-1, 1].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_learning_curves.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig1_learning_curves.png"), bbox_inches="tight")
    plt.close()
    print("Generated: fig1_learning_curves.pdf")


def figure_2_final_performance(df: pd.DataFrame, output_dir: str):
    """
    Figure 2: Bar chart of final performance (success rate) with error bars.
    """
    # Aggregate by env and algo
    summary = (
        df.groupby(["env", "algo"])
        .agg(
            {
                "final_100_success_rate": ["mean", "std"],
                "mean_length": ["mean", "std"],
            }
        )
        .reset_index()
    )

    summary.columns = ["env", "algo", "success_mean", "success_std", "length_mean", "length_std"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Success rate
    ax = axes[0]
    envs = sorted(summary["env"].unique())
    x = np.arange(len(envs))
    width = 0.25

    for i, algo in enumerate(["PPO", "DQN", "A2C"]):
        algo_data = summary[summary["algo"] == algo].set_index("env")
        means = [algo_data.loc[e, "success_mean"] if e in algo_data.index else 0 for e in envs]
        stds = [algo_data.loc[e, "success_std"] if e in algo_data.index else 0 for e in envs]

        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=algo,
            color=COLORS[algo],
            capsize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Success Rate")
    ax.set_title("Final Performance: Success Rate (last 100 episodes)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Episode length
    ax = axes[1]
    for i, algo in enumerate(["PPO", "DQN", "A2C"]):
        algo_data = summary[summary["algo"] == algo].set_index("env")
        means = [algo_data.loc[e, "length_mean"] if e in algo_data.index else 0 for e in envs]
        stds = [algo_data.loc[e, "length_std"] if e in algo_data.index else 0 for e in envs]

        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=algo,
            color=COLORS[algo],
            capsize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Episode Length")
    ax.set_title("Final Performance: Average Episode Length")
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_final_performance.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig2_final_performance.png"), bbox_inches="tight")
    plt.close()
    print("Generated: fig2_final_performance.pdf")


def figure_3_phase_completion(df: pd.DataFrame, output_dir: str):
    """
    Figure 3: Key pickup and door open rates for E3, E4, E6.
    Shows how well agents learn subtasks.
    """
    keydoor_envs = ["E3", "E4", "E6"]
    keydoor_df = df[df["env"].isin(keydoor_envs)]

    if keydoor_df.empty:
        print("No key-door environments found, skipping figure 3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    summary = (
        keydoor_df.groupby(["env", "algo"])
        .agg(
            {
                "key_pickup_rate": ["mean", "std"],
                "door_open_rate": ["mean", "std"],
            }
        )
        .reset_index()
    )
    summary.columns = ["env", "algo", "key_mean", "key_std", "door_mean", "door_std"]

    envs = sorted(keydoor_df["env"].unique())
    x = np.arange(len(envs))
    width = 0.25

    # Key pickup
    ax = axes[0]
    for i, algo in enumerate(["PPO", "DQN", "A2C"]):
        algo_data = summary[summary["algo"] == algo].set_index("env")
        means = [algo_data.loc[e, "key_mean"] if e in algo_data.index else 0 for e in envs]
        stds = [algo_data.loc[e, "key_std"] if e in algo_data.index else 0 for e in envs]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=algo,
            color=COLORS[algo],
            capsize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Key Pickup Rate")
    ax.set_title("Phase 1 Completion: Key Pickup")
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Door open
    ax = axes[1]
    for i, algo in enumerate(["PPO", "DQN", "A2C"]):
        algo_data = summary[summary["algo"] == algo].set_index("env")
        means = [algo_data.loc[e, "door_mean"] if e in algo_data.index else 0 for e in envs]
        stds = [algo_data.loc[e, "door_std"] if e in algo_data.index else 0 for e in envs]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=algo,
            color=COLORS[algo],
            capsize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Door Open Rate")
    ax.set_title("Phase 2 Completion: Door Open")
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_phase_completion.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig3_phase_completion.png"), bbox_inches="tight")
    plt.close()
    print("Generated: fig3_phase_completion.pdf")


def figure_4_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Figure 4: Heatmap of success rates (env x algo).
    """
    if len(df["env"].unique()) < 2 or len(df["algo"].unique()) < 2:
        print("Skipping heatmap: need at least 2 environments and 2 algorithms")
        return

    pivot = df.groupby(["env", "algo"])["final_100_success_rate"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Success Rate"},
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Environment")
    ax.set_title("Success Rate: Environment × Algorithm")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig4_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("Generated: fig4_heatmap.pdf")


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Generate LaTeX-ready summary table."""
    summary = (
        df.groupby(["env", "algo"])
        .agg(
            {
                "final_100_success_rate": ["mean", "std"],
                "mean_reward": ["mean", "std"],
                "mean_length": ["mean", "std"],
            }
        )
        .round(3)
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Save CSV
    csv_path = os.path.join(output_dir, "results_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Generated: {csv_path}")

    # Generate LaTeX
    latex_path = os.path.join(output_dir, "results_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experiment Results Summary}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Env & Algo & Success Rate & Reward & Length \\\\\n")
        f.write("\\midrule\n")

        for _, row in summary.iterrows():
            success = f"{row['final_100_success_rate_mean']:.2f} ± {row['final_100_success_rate_std']:.2f}"
            reward = f"{row['mean_reward_mean']:.1f} ± {row['mean_reward_std']:.1f}"
            length = f"{row['mean_length_mean']:.1f} ± {row['mean_length_std']:.1f}"
            f.write(f"{row['env']} & {row['algo']} & {success} & {reward} & {length} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\end{table}\n")

    print(f"Generated: {latex_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/models")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading experiments...")
    df = load_all_experiments(args.results_dir)

    if df is None or df.empty:
        print("No experiments found. Run some training first!")
        return

    print(f"Found {len(df)} experiment runs")
    print(f"Environments: {sorted(df['env'].unique())}")
    print(f"Algorithms: {sorted(df['algo'].unique())}")

    print("\nGenerating figures...")
    figure_1_learning_curves(df, args.output_dir)
    figure_2_final_performance(df, args.output_dir)
    figure_3_phase_completion(df, args.output_dir)
    figure_4_heatmap(df, args.output_dir)

    print("\nGenerating tables...")
    generate_summary_table(df, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
