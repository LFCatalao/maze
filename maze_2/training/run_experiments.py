"""
Run full experiment matrix with multiple seeds.
"""

import os
import itertools
import subprocess
from datetime import datetime


ENVIRONMENTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
ALGORITHMS = ["PPO", "DQN", "A2C"]
SEEDS = [42, 123, 456, 789, 1024]

# Timesteps per environment complexity
TIMESTEPS = {
    "E1": 100_000,
    "E2": 150_000,
    "E3": 200_000,
    "E4": 250_000,
    "E5": 200_000,
    "E6": 300_000,
}


def run_experiment(env: str, algo: str, seed: int):
    """Run single experiment."""
    steps = TIMESTEPS[env]

    cmd = [
        "python",
        "training/train_experiment.py",
        "--env",
        env,
        "--reward",
        "simple",
        "--algo",
        algo,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {env} + {algo} (seed={seed})")
    print(f"{'='*60}")

    subprocess.run(cmd)


def run_all(envs=None, algos=None, seeds=None):
    """Run full or partial experiment matrix."""
    envs = envs or ENVIRONMENTS
    algos = algos or ALGORITHMS
    seeds = seeds or SEEDS

    total = len(envs) * len(algos) * len(seeds)
    current = 0

    start_time = datetime.now()

    for env, algo, seed in itertools.product(envs, algos, seeds):
        current += 1
        print(f"\n[{current}/{total}] Starting experiment...")
        run_experiment(env, algo, seed)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED {total} experiments in {elapsed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None, help="Environments to run")
    parser.add_argument("--algos", nargs="+", default=None, help="Algorithms to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run")
    parser.add_argument("--quick", action="store_true", help="Quick test: E1+E3, PPO only, 1 seed")
    args = parser.parse_args()

    if args.quick:
        run_all(envs=["E1", "E3"], algos=["PPO"], seeds=[42])
    else:
        run_all(envs=args.envs, algos=args.algos, seeds=args.seeds)
