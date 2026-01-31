"""
Quick test: Train PPO briefly on all environments, generate metrics.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from environments import LockedRoomEnv, get_config, apply_reward_wrapper
from training.train_experiment import FlattenObservation


class QuickMetricsCallback(BaseCallback):
    """Lightweight callback for quick testing."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.current_reward = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1

        if self.locals.get("dones", [False])[0]:
            success = self.current_reward >= 90
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.successes.append(1 if success else 0)
            self.current_reward = 0
            self.current_length = 0

        return True

    def get_summary(self):
        if not self.episode_rewards:
            return {"episodes": 0}
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "success_rate": float(np.mean(self.successes)),
            "final_10_success": (
                float(np.mean(self.successes[-10:])) if len(self.successes) >= 10 else 0
            ),
        }


def create_env(env_id: str):
    """Create wrapped environment."""
    config = get_config(env_id)
    env = LockedRoomEnv(
        size=config.size,
        observation_mode=config.observation_mode,
        render_mode=None,
        max_steps=config.max_steps,
        fixed_agent_pos=config.fixed_agent_pos,
        fixed_goal_pos=config.fixed_goal_pos,
        num_doors=config.num_doors,
        randomize_doors=config.randomize_doors,
        include_key=config.include_key,
        locked_door=config.locked_door,
    )
    env = apply_reward_wrapper(env, "simple")
    env = FlattenObservation(env)
    return env


def quick_test(
    environments=None,
    timesteps=10_000,
    seed=42,
    output_dir="results/quick_test",
):
    """
    Run quick PPO training on specified environments.

    Args:
        environments: List of env IDs, or None for all
        timesteps: Training steps per environment (default 10k for speed)
        seed: Random seed
        output_dir: Where to save results
    """

    if environments is None:
        environments = ["E1", "E2", "E3", "E4", "E5", "E6"]

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    print("=" * 70)
    print("QUICK TEST: PPO on all environments")
    print(f"Timesteps per env: {timesteps:,}")
    print(f"Seed: {seed}")
    print("=" * 70)

    for env_id in environments:
        print(f"\n--- Training {env_id} ---")

        # Create environment
        env = DummyVecEnv([lambda eid=env_id: create_env(eid)])

        # Create callback
        callback = QuickMetricsCallback()

        # Create and train model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            ent_coef=0.1,
            seed=seed,
        )

        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

        # Save model
        model_path = os.path.join(output_dir, f"{env_id}_PPO_quick")
        model.save(model_path)

        # Get summary
        summary = callback.get_summary()
        results[env_id] = summary

        print(f"  Episodes: {summary['episodes']}")
        print(f"  Mean reward: {summary['mean_reward']:.1f}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Model saved: {model_path}.zip")

        env.close()

    # Save combined results
    results_path = os.path.join(output_dir, "quick_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Env':<8} {'Episodes':>10} {'Reward':>10} {'Success':>10}")
    print("-" * 40)
    for env_id, summary in results.items():
        print(
            f"{env_id:<8} {summary['episodes']:>10} {summary['mean_reward']:>10.1f} {summary['success_rate']:>9.1%}"
        )

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return results


def evaluate_models(model_dir="results/quick_test", episodes=20):
    """
    Load and evaluate saved models.
    """
    print("\n" + "=" * 70)
    print("EVALUATING SAVED MODELS")
    print("=" * 70)

    from stable_baselines3 import PPO

    results = {}

    for model_file in sorted(os.listdir(model_dir)):
        if not model_file.endswith(".zip"):
            continue

        env_id = model_file.split("_")[0]
        model_path = os.path.join(model_dir, model_file)

        print(f"\nEvaluating {env_id}...")

        # Load model
        model = PPO.load(model_path)

        # Create env
        env = create_env(env_id)

        # Run episodes
        successes = []
        rewards = []
        lengths = []

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            successes.append(1 if total_reward >= 90 else 0)
            rewards.append(total_reward)
            lengths.append(steps)

        results[env_id] = {
            "success_rate": np.mean(successes),
            "mean_reward": np.mean(rewards),
            "mean_length": np.mean(lengths),
        }

        print(
            f"  Success: {np.mean(successes):.1%}, Reward: {np.mean(rewards):.1f}, Length: {np.mean(lengths):.1f}"
        )

        env.close()

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Env':<8} {'Success':>10} {'Reward':>10} {'Length':>10}")
    print("-" * 40)
    for env_id, data in results.items():
        print(
            f"{env_id:<8} {data['success_rate']:>9.1%} {data['mean_reward']:>10.1f} {data['mean_length']:>10.1f}"
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick PPO test across all environments")
    parser.add_argument(
        "--envs", nargs="+", default=None, help="Environments to test (default: all)"
    )
    parser.add_argument("--timesteps", type=int, default=10_000, help="Timesteps per environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results/quick_test", help="Output directory")
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Only evaluate existing models"
    )
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_models(model_dir=args.output)
    else:
        quick_test(
            environments=args.envs,
            timesteps=args.timesteps,
            seed=args.seed,
            output_dir=args.output,
        )
        print("\nRunning evaluation...")
        evaluate_models(model_dir=args.output)
