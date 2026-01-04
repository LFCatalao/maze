"""
Plot results from quick_test.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_quick_test(results_file="results/quick_test/quick_test_results.json"):
    with open(results_file) as f:
        results = json.load(f)

    envs = list(results.keys())
    success_rates = [results[e].get("success_rate", 0) for e in envs]
    rewards = [results[e].get("mean_reward", 0) for e in envs]
    lengths = [results[e].get("mean_length", 0) for e in envs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Success rate
    ax = axes[0]
    ax.bar(envs, success_rates, color="#2ecc71")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Environment")
    ax.set_ylim(0, 1)

    # Reward
    ax = axes[1]
    ax.bar(envs, rewards, color="#3498db")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward by Environment")

    # Episode length
    ax = axes[2]
    ax.bar(envs, lengths, color="#e74c3c")
    ax.set_ylabel("Episode Length")
    ax.set_title("Mean Episode Length")

    plt.suptitle("Quick Test Results (PPO, 10k steps)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/quick_test/quick_test_figures.png", dpi=150)
    plt.show()

    # Print summary
    print("\nQUICK TEST SUMMARY")
    print("=" * 50)
    print(f"{'Env':<6} {'Success':>10} {'Reward':>10} {'Length':>10}")
    print("-" * 50)
    for env in envs:
        r = results[env]
        print(
            f"{env:<6} {r.get('success_rate', 0):>9.1%} {r.get('mean_reward', 0):>10.1f} {r.get('mean_length', 0):>10.1f}"
        )


if __name__ == "__main__":
    plot_quick_test()
