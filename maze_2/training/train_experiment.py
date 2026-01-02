"""
Main Training Script for RL Comparison Study

Usage:
    python train_experiment.py --env E1 --reward R2 --algo PPO --obs full_map --steps 100000
    python train_experiment.py --list  # List all options
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments import (
    LockedRoomEnv,
    get_config,
    apply_reward_wrapper,
    list_configs,
    ALL_ENV_CONFIGS,
    REWARD_WRAPPERS,
)


class FlattenObservation(gym.ObservationWrapper):
    """Simple observation with key support."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 20, (12,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_y, agent_x = base_env.agent_pos
        goal_y, goal_x = base_env.goal_pos
        direction = base_env.agent_dir

        has_key = 1.0 if base_env.carrying is not None else 0.0

        key_y, key_x = -1.0, -1.0
        on_key = 0.0
        if hasattr(base_env, "key_positions") and base_env.key_positions:
            if base_env.carrying is None:
                key_pos = list(base_env.key_positions.keys())[0]
                key_y, key_x = float(key_pos[0]), float(key_pos[1])
                if agent_y == key_pos[0] and agent_x == key_pos[1]:
                    on_key = 1.0

        door_y, door_x = -1.0, -1.0
        on_door = 0.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_y, door_x = float(door_pos[0]), float(door_pos[1])
            dist_to_door = abs(agent_y - door_pos[0]) + abs(agent_x - door_pos[1])
            if dist_to_door <= 1:
                on_door = 1.0

        return np.array(
            [
                agent_y,
                agent_x,
                goal_y,
                goal_x,
                direction,
                has_key,
                key_y,
                key_x,
                door_y,
                door_x,
                on_key,
                on_door,
            ],
            dtype=np.float32,
        )


def create_env(env_id: str, reward_id: str, obs_mode: str, render_mode: Optional[str] = None):
    """Create environment with specified configuration"""

    # Get environment config
    config = get_config(env_id, obs_mode)

    # Create base environment
    env = LockedRoomEnv(
        size=config.size,
        observation_mode=config.observation_mode,
        render_mode=render_mode,
        max_steps=config.max_steps,
        fixed_agent_pos=config.fixed_agent_pos,
        fixed_goal_pos=config.fixed_goal_pos,
        num_doors=config.num_doors,
        randomize_doors=config.randomize_doors,
        include_key=config.include_key,
        locked_door=config.locked_door,
        verbose=False,
    )

    # Apply reward wrapper
    env = apply_reward_wrapper(env, reward_id)

    return env


def train(
    env_id: str = "E1",
    reward_id: str = "R2",
    algo: str = "PPO",
    obs_mode: str = "full_map",
    total_timesteps: int = 100_000,
    checkpoint_freq: int = 25_000,
    save_dir: str = "results/models",
    log_dir: str = "results/logs",
    seed: int = 42,
):
    """
    Train an agent with specified configuration.

    Args:
        env_id: Environment configuration (E1, E2, E3, E4)
        reward_id: Reward strategy (R1, R2, R3, R4)
        algo: Algorithm (PPO, DQN, A2C, SAC)
        obs_mode: Observation mode (full_map, partial_view, agent_view)
        total_timesteps: Total training steps
        checkpoint_freq: Save checkpoint every N steps
        save_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        seed: Random seed
    """

    # Create experiment name
    exp_name = f"{env_id}_{reward_id}_{algo}_{obs_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{timestamp}"

    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"  Environment: {env_id}")
    print(f"  Reward: {reward_id}")
    print(f"  Algorithm: {algo}")
    print(f"  Observation: {obs_mode}")
    print(f"  Total steps: {total_timesteps:,}")
    print(f"  Seed: {seed}")
    print("=" * 70)

    # Import algorithm
    try:
        if algo == "PPO":
            from stable_baselines3 import PPO as Algorithm
        elif algo == "DQN":
            from stable_baselines3 import DQN as Algorithm
        elif algo == "A2C":
            from stable_baselines3 import A2C as Algorithm
        elif algo == "SAC":
            # SAC requires continuous action space - skip for now
            print("ERROR: SAC requires continuous actions. Use PPO, DQN, or A2C.")
            return None
        else:
            print(f"ERROR: Unknown algorithm: {algo}")
            return None
    except ImportError:
        print("ERROR: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        return None

    # Create directories
    model_dir = os.path.join(save_dir, run_name)
    tb_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"\nModel directory: {model_dir}")
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"\nTo monitor: tensorboard --logdir={log_dir}")

    # Create environment
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    def make_env():
        env = create_env(env_id, reward_id, obs_mode, render_mode=None)
        env = FlattenObservation(env)
        return env

    env = DummyVecEnv([make_env])

    print(f"\nEnvironment created:")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    # Create callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=model_dir,
        name_prefix=exp_name,
    )

    # Create model with appropriate hyperparameters
    if algo == "PPO":
        model = Algorithm(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.1,  # Entropy for exploration
            tensorboard_log=tb_log_dir,
            seed=seed,
        )
    elif algo == "DQN":
        model = Algorithm(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            tensorboard_log=tb_log_dir,
            seed=seed,
        )
    elif algo == "A2C":
        model = Algorithm(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.01,
            tensorboard_log=tb_log_dir,
            seed=seed,
        )

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")

    # Save final model
    final_path = os.path.join(model_dir, f"{exp_name}_final")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}.zip")

    # Save experiment config
    config_path = os.path.join(model_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Environment: {env_id}\n")
        f.write(f"Reward: {reward_id}\n")
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Observation: {obs_mode}\n")
        f.write(f"Total steps: {total_timesteps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Timestamp: {timestamp}\n")

    print(f"Config saved: {config_path}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    env.close()
    return model, final_path


def list_options():
    """List all available options"""
    print("\n" + "=" * 70)
    print("AVAILABLE OPTIONS")
    print("=" * 70)

    print("\n--- Environment Configurations ---")
    list_configs()

    print("\n--- Reward Strategies ---")
    print("  simple - Distance + step penalty + room bonus")

    print("\n--- Algorithms ---")
    print("  PPO  - Proximal Policy Optimization")
    print("  DQN  - Deep Q-Network")
    print("  A2C  - Advantage Actor-Critic")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents for comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_experiment.py --env E1 --reward R2 --algo PPO
  python train_experiment.py --env E1 --reward R4 --algo DQN --steps 200000
  python train_experiment.py --list
        """,
    )

    parser.add_argument(
        "--env",
        type=str,
        default="E1",
        choices=list(ALL_ENV_CONFIGS.keys()),
        help="Environment configuration",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="R2",
        choices=list(REWARD_WRAPPERS.keys()),
        help="Reward strategy",
    )
    parser.add_argument(
        "--algo", type=str, default="PPO", choices=["PPO", "DQN", "A2C"], help="RL algorithm"
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="full_map",
        choices=["full_map", "partial_view", "agent_view"],
        help="Observation mode",
    )
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000, help="Checkpoint frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list", action="store_true", help="List all available options")

    args = parser.parse_args()

    if args.list:
        list_options()
        return

    train(
        env_id=args.env,
        reward_id=args.reward,
        algo=args.algo,
        obs_mode=args.obs,
        total_timesteps=args.steps,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
