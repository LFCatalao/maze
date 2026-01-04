#!/usr/bin/env python3
"""
Evaluation Script - Watch trained models in action

Usage:
    python evaluate.py <model_path>                    # Basic evaluation
    python evaluate.py <model_path> --episodes 20     # More episodes
    python evaluate.py <model_path> --no-render       # Fast evaluation without visuals
    python evaluate.py --random                       # Watch random agent (no model needed)
    python evaluate.py --manual                       # Manual keyboard control
"""

import argparse
import os
import sys
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments import (
    LockedRoomEnv,
    get_config,
    apply_reward_wrapper,
    ALL_ENV_CONFIGS,
    REWARD_WRAPPERS,
)

import gymnasium as gym
from gymnasium import spaces


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


def create_env(env_id: str, reward_id: str, obs_mode: str, render: bool = True):
    """Create environment with specified configuration"""
    config = get_config(env_id, obs_mode)

    env = LockedRoomEnv(
        size=config.size,
        observation_mode=config.observation_mode,
        render_mode="human" if render else None,
        max_steps=config.max_steps,
        fixed_agent_pos=config.fixed_agent_pos,
        fixed_goal_pos=config.fixed_goal_pos,
        num_doors=config.num_doors,
        randomize_doors=config.randomize_doors,
        include_key=config.include_key,
        locked_door=config.locked_door,
        defined_doors=config.defined_doors,
        goal_in_locked_room=config.goal_in_locked_room,
        enable_key_chain=config.enable_key_chain,
        verbose=True,  # Show action feedback
    )

    env = apply_reward_wrapper(env, reward_id)
    return env


def evaluate_model(
    model_path: str,
    env_id: str = "E1",
    reward_id: str = "R2",
    obs_mode: str = "full_map",
    num_episodes: int = 10,
    render: bool = True,
    delay_ms: int = 100,
):
    """
    Evaluate a trained model

    Args:
        model_path: Path to saved model (without .zip extension)
        env_id: Environment configuration
        reward_id: Reward strategy
        obs_mode: Observation mode
        num_episodes: Number of episodes to run
        render: Whether to show pygame visualization
        delay_ms: Delay between steps in milliseconds (for visualization)
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load model
    try:
        from stable_baselines3 import PPO, DQN, A2C
    except ImportError:
        print("ERROR: stable-baselines3 not installed")
        return

    # Auto-detect algorithm from filename or try each
    model = None
    for AlgoClass in [PPO, DQN, A2C]:
        try:
            model = AlgoClass.load(model_path)
            print(f"Loaded model with {AlgoClass.__name__}")
            break
        except:
            continue

    if model is None:
        print(f"ERROR: Could not load model from {model_path}")
        print("Make sure the file exists and has .zip extension")
        return

    # Create environment
    env = create_env(env_id, reward_id, obs_mode, render=render)
    env_flat = FlattenObservation(env)

    print(f"\nConfiguration:")
    print(f"  Environment: {env_id}")
    print(f"  Reward: {reward_id}")
    print(f"  Observation: {obs_mode}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Render: {render}")
    print("=" * 70)

    # Action names for logging
    action_names = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "PICKUP", 4: "TOGGLE"}

    # Run evaluation
    successes = 0
    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        obs, info = env_flat.reset()
        done = False
        episode_reward = 0
        steps = 0

        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Agent start: {info['agent_pos']}, Goal: {info['goal_pos']}")
        print(f"Mission: {info['mission']}")
        print(f"{'='*50}")

        while not done:
            if render:
                env.render()

                # Handle pygame events
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Step
            prev_pos = info["agent_pos"].copy()
            obs, reward, terminated, truncated, info = env_flat.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # Log step
            moved = prev_pos != info["agent_pos"]
            status = "MOVED" if moved else "no move"
            if (
                steps <= 20 or reward > 0.1 or terminated
            ):  # Log first 20 steps and important events
                print(
                    f"  Step {steps}: {action_names[action]:8} | pos={info['agent_pos']} | reward={reward:+.3f} | {status}"
                )

            if render and delay_ms > 0:
                time.sleep(delay_ms / 1000)

        total_rewards.append(episode_reward)
        total_steps.append(steps)

        if terminated and episode_reward > 0:
            successes += 1
            print(f"\n  ✓✓✓ SUCCESS! Reached goal in {steps} steps ✓✓✓")
        else:
            print(f"\n  ✗ Failed (truncated after {steps} steps)")

        print(f"  Episode reward: {episode_reward:.3f}")

        if render:
            time.sleep(0.5)  # Pause between episodes

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"\nReward Statistics:")
    print(f"  Mean:  {np.mean(total_rewards):.3f}")
    print(f"  Std:   {np.std(total_rewards):.3f}")
    print(f"  Min:   {np.min(total_rewards):.3f}")
    print(f"  Max:   {np.max(total_rewards):.3f}")
    print(f"\nStep Statistics:")
    print(f"  Mean:  {np.mean(total_steps):.1f}")
    print(f"  Min:   {np.min(total_steps)}")
    print(f"  Max:   {np.max(total_steps)}")

    if successes > 0:
        successful_steps = [total_steps[i] for i in range(num_episodes) if total_rewards[i] > 0]
        print(f"\nSuccessful Episodes:")
        print(f"  Mean steps: {np.mean(successful_steps):.1f}")

    print("=" * 70)

    env.close()


def run_random_agent(
    env_id: str = "E1", reward_id: str = "simple", num_episodes: int = 3, delay_ms: int = 150
):
    """Run random agent for visualization/debugging"""
    print("=" * 70)
    print("RANDOM AGENT")
    print("=" * 70)

    env = create_env(env_id, reward_id, "full_map", render=True)
    action_names = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "PICKUP", 4: "TOGGLE"}

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}")
        print(f"Agent: {info['agent_pos']}, Goal: {info['goal_pos']}")

        while not done and steps < 200:
            env.render()

            import pygame

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            if reward != 0:
                print(f"  Step {steps}: {action_names[action]} -> reward={reward:.3f}")

            time.sleep(delay_ms / 1000)

        print(f"Episode finished: {steps} steps, reward={episode_reward:.3f}")
        if terminated:
            print("  ✓ Goal reached!")

    env.close()


def manual_control(env_id: str = "E1", reward_id: str = "simple"):
    """Manual keyboard control for testing"""
    print("=" * 70)
    print("MANUAL CONTROL")
    print("=" * 70)
    print("\nControls:")
    print("  ← → : Turn left/right")
    print("  ↑   : Move forward")
    print("  SPACE: Pick up")
    print("  E   : Toggle (open door)")
    print("  R   : Reset")
    print("  ESC : Quit")
    print("=" * 70)

    import pygame

    env = create_env(env_id, reward_id, "full_map", render=True)
    obs, info = env.reset()

    print(f"\nAgent: {info['agent_pos']}, Goal: {info['goal_pos']}")
    print(f"Mission: {info['mission']}")

    running = True
    total_reward = 0
    steps = 0

    while running:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                action = None

                if event.key == pygame.K_LEFT:
                    action = 0  # Turn left
                elif event.key == pygame.K_RIGHT:
                    action = 1  # Turn right
                elif event.key == pygame.K_UP:
                    action = 2  # Forward
                elif event.key == pygame.K_SPACE:
                    action = 3  # Pickup
                elif event.key == pygame.K_e:
                    action = 4  # Toggle
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    steps = 0
                    print(f"\n--- RESET ---")
                    print(f"Agent: {info['agent_pos']}, Goal: {info['goal_pos']}")
                elif event.key == pygame.K_ESCAPE:
                    running = False

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1

                    if reward != 0:
                        print(f"Step {steps}: reward={reward:.3f}, total={total_reward:.3f}")

                    if terminated:
                        print(f"\n✓✓✓ GOAL REACHED in {steps} steps! ✓✓✓")
                        print(f"Total reward: {total_reward:.3f}")
                        print("Press R to reset or ESC to quit")

        pygame.time.wait(30)

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py results/models/E1_R2_PPO_full_map_*/E1_R2_PPO_full_map_final
  python evaluate.py my_model --episodes 20
  python evaluate.py my_model --no-render
  python evaluate.py --random
  python evaluate.py --manual
        """,
    )

    parser.add_argument(
        "model_path", nargs="?", default=None, help="Path to model file (without .zip)"
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
        default="simple",
        choices=list(REWARD_WRAPPERS.keys()),
        help="Reward strategy",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="full_map",
        choices=["full_map", "partial_view", "agent_view"],
        help="Observation mode",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization (faster)")
    parser.add_argument("--delay", type=int, default=100, help="Delay between steps in ms")
    parser.add_argument("--random", action="store_true", help="Run random agent instead of model")
    parser.add_argument("--manual", action="store_true", help="Manual keyboard control")

    args = parser.parse_args()

    if args.manual:
        manual_control(args.env, args.reward)
    elif args.random:
        run_random_agent(args.env, args.reward, args.episodes, args.delay)
    elif args.model_path:
        evaluate_model(
            model_path=args.model_path,
            env_id=args.env,
            reward_id=args.reward,
            obs_mode=args.obs,
            num_episodes=args.episodes,
            render=not args.no_render,
            delay_ms=args.delay,
        )
    else:
        print("ERROR: Provide a model path, or use --random or --manual")
        print("\nExamples:")
        print("  python evaluate.py path/to/model")
        print("  python evaluate.py --random")
        print("  python evaluate.py --manual")
        parser.print_help()


if __name__ == "__main__":
    main()
