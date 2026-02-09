#!/usr/bin/env python3
"""
Evaluation Script - Watch trained models in action

Usage:
    python evaluate.py <model_path>                    # Basic evaluation (fog of war ON)
    python evaluate.py <model_path> --full-map        # Show full map (fog of war OFF)
    python evaluate.py <model_path> --episodes 20     # More episodes
    python evaluate.py <model_path> --no-render       # Fast evaluation without visuals
    python evaluate.py --random                       # Watch random agent (no model needed)
    python evaluate.py --manual                       # Manual keyboard control

Fog of War:
    By default, only shows what the agent has explored (fog of war enabled).
    Use --full-map to see the entire environment.
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
    LegacySimpleObs,
    SimpleObs,
    EnhancedObs,
)

# Import hyperparameters for display
from environments.reward_wrappers import (
    REWARD_GOAL,
    REWARD_CLOSER,
    REWARD_CLOSER_TO_EXIT,
    PENALTY_STEP,
    PENALTY_NO_MOVE,
    REWARD_KEY_PICKUP,
    REWARD_DOOR_OPEN,
    REWARD_EXPLORATION,
    PENALTY_REVISIT,
    PENALTY_DIRECTION_CHANGE,
    PENALTY_REVERSAL,
    REWARD_USEFUL_ROOM,
    PENALTY_EMPTY_ROOM,
)

import gymnasium as gym
from gymnasium import spaces


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
        use_fixed_key_positions=config.use_fixed_key_positions,
        verbose=True,  # Show action feedback
    )

    env = apply_reward_wrapper(env, reward_id)
    return env


def evaluate_model(
    model_path: str,
    env_id: str = None,
    reward_id: str = None,
    obs_mode: str = None,
    num_episodes: int = 10,
    render: bool = True,
    delay_ms: int = 100,
    fog_of_war: bool = True,
):
    """
    Evaluate a trained model

    Args:
        model_path: Path to saved model (without .zip extension)
        env_id: Environment configuration (auto-detected from filename if None)
        reward_id: Reward strategy (auto-detected from filename if None)
        obs_mode: Observation mode (auto-detected from filename if None)
        num_episodes: Number of episodes to run
        render: Whether to show pygame visualization
        delay_ms: Delay between steps in milliseconds (for visualization)
        fog_of_war: Only show explored areas (what agent has seen)
    """
    # Auto-detect configuration from model filename if not specified
    if env_id is None or reward_id is None or obs_mode is None:
        import re
        filename = os.path.basename(model_path)
        
        # Try to extract env, reward, and obs from filename
        # Expected format: E<X>_<reward>_<algo>_<obs>_<timestamp>_final or E<X>_FIXED_<reward>_...
        # First try to match with FIXED/RANDOM suffix
        match = re.match(r'(E\d+(?:_(?:FIXED|RANDOM))?)[_\-](\w+)[_\-](?:PPO|DQN|A2C)[_\-](full_map|partial_view|agent_view)', filename)
        if match:
            detected_env, detected_reward, detected_obs = match.groups()
            if env_id is None:
                env_id = detected_env
                print(f"Auto-detected environment: {env_id}")
            if reward_id is None:
                reward_id = detected_reward
                print(f"Auto-detected reward: {reward_id}")
            if obs_mode is None:
                obs_mode = detected_obs
                print(f"Auto-detected observation mode: {obs_mode}")
    
    # Set defaults if still None
    if env_id is None:
        env_id = "E7"
    if reward_id is None:
        reward_id = "simple"
    if obs_mode is None:
        obs_mode = "full_map"
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load model
    try:
        from stable_baselines3 import PPO, DQN, A2C
    except ImportError as e:
        print("ERROR: stable-baselines3 not installed or incompatible")
        print(f"Details: {e}")
        print("\nIf you see NumPy 2.x errors, downgrade NumPy:")
        print("  pip install 'numpy<2'")
        return
    except RuntimeError as e:
        print("ERROR: PyTorch/NumPy compatibility issue")
        print(f"Details: {e}")
        print("\nTo fix, downgrade NumPy:")
        print("  pip install 'numpy<2'")
        return

    # Auto-detect algorithm from filename or try each
    model = None
    load_errors = []
    for AlgoClass in [PPO, DQN, A2C]:
        try:
            model = AlgoClass.load(model_path)
            print(f"Loaded model with {AlgoClass.__name__}")
            break
        except Exception as e:
            load_errors.append(f"{AlgoClass.__name__}: {str(e)[:100]}")
            continue

    if model is None:
        print(f"ERROR: Could not load model from {model_path}")
        print("Make sure the file exists and has .zip extension")
        print("\nTried loading with:")
        for err in load_errors:
            print(f"  - {err}")
        return

    # Create environment
    env = create_env(env_id, reward_id, obs_mode, render=render)
    
    # Auto-detect which observation wrapper to use based on model's observation space
    model_obs_shape = model.observation_space.shape[0]
    
    if model_obs_shape == 10:
        # Legacy simple observation (10 features: agent, goal, has_key, key_pos, door_pos, door_locked)
        env_flat = LegacySimpleObs(env)
        obs_wrapper_name = "LegacySimpleObs (10 features)"
    elif model_obs_shape == 34:
        # Simple observation (34 features: agent, goal, carrying, 4 keys, 4 doors)
        env_flat = SimpleObs(env)
        obs_wrapper_name = "SimpleObs (34 features)"
    elif model_obs_shape == 1233:
        # Enhanced observation (1233 features: includes spatial features)
        env_flat = EnhancedObs(env)
        obs_wrapper_name = "EnhancedObs (1233 features)"
    else:
        print(f"ERROR: Unknown observation space size: {model_obs_shape}")
        print("Expected 10 (LegacySimpleObs), 34 (SimpleObs), or 1233 (EnhancedObs)")
        env.close()
        return
    
    # Get base environment for rendering
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    print(f"\nConfiguration:")
    print(f"  Environment: {env_id}")
    print(f"  Reward: {reward_id}")
    print(f"  Observation: {obs_mode}")
    print(f"  Observation wrapper: {obs_wrapper_name}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Render: {render}")
    print(f"  Fog of War: {fog_of_war} {'(showing only explored areas)' if fog_of_war else '(showing full map)'}")
    
    # Display hyperparameters
    print(f"\nReward Hyperparameters:")
    print(f"  PENALTY_STEP: {PENALTY_STEP}")
    print(f"  PENALTY_NO_MOVE: {PENALTY_NO_MOVE}")
    print(f"  REWARD_EXPLORATION: {REWARD_EXPLORATION} (per new cell)")
    print(f"  REWARD_KEY_PICKUP: {REWARD_KEY_PICKUP}")
    print(f"  REWARD_DOOR_OPEN: {REWARD_DOOR_OPEN}")
    print(f"  REWARD_CLOSER: {REWARD_CLOSER} (new best distance)")
    print(f"  REWARD_CLOSER_TO_EXIT: {REWARD_CLOSER_TO_EXIT} (empty room exit)")
    print(f"  PENALTY_REVISIT: {PENALTY_REVISIT} (per revisit)")
    print(f"  PENALTY_DIRECTION_CHANGE: {PENALTY_DIRECTION_CHANGE}")
    print(f"  PENALTY_REVERSAL: {PENALTY_REVERSAL} (180° turn)")
    print(f"  REWARD_USEFUL_ROOM: {REWARD_USEFUL_ROOM}")
    print(f"  PENALTY_EMPTY_ROOM: {PENALTY_EMPTY_ROOM}")
    print(f"  REWARD_GOAL: {REWARD_GOAL}")
    print("=" * 70)

    # Action names for logging (omnidirectional movement)
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

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
                base_env.render(fog_of_war=fog_of_war)

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
            try:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            except RuntimeError as e:
                if "Could not infer dtype" in str(e):
                    print("\n" + "=" * 70)
                    print("ERROR: NumPy compatibility issue detected!")
                    print("=" * 70)
                    print("Your model was trained with NumPy 1.x but you have NumPy 2.x installed.")
                    print("\nTo fix this, downgrade NumPy:")
                    print("  pip install 'numpy<2'")
                    print("\nThen try running the evaluation again.")
                    print("=" * 70)
                    env.close()
                    return
                else:
                    raise

            # Step
            prev_pos = info["agent_pos"].copy()
            obs, reward, terminated, truncated, info = env_flat.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # Log step
            moved = not np.array_equal(prev_pos, info["agent_pos"])
            status = "MOVED" if moved else "no move"
            # Print all steps with reward breakdown if available
            reward_str = f"reward={reward:+.3f}"
            if 'reward_breakdown' in info:
                breakdown = info['reward_breakdown']
                breakdown_str = " | ".join([f"{k}={v:+.2f}" for k, v in breakdown.items() if v != 0])
                if breakdown_str:
                    reward_str += f" ({breakdown_str})"
            print(
                f"  Step {steps}: {action_names[action]:8} | pos={info['agent_pos']} | {reward_str} | {status}"
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
    env_id: str = "E7", reward_id: str = "simple", num_episodes: int = 3, delay_ms: int = 150, fog_of_war: bool = True
):
    """Run random agent for visualization/debugging"""
    print("=" * 70)
    print("RANDOM AGENT")
    print("=" * 70)

    env = create_env(env_id, reward_id, "full_map", render=True)
    
    # Get base environment for rendering
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}")
        print(f"Agent: {info['agent_pos']}, Goal: {info['goal_pos']}")

        while not done and steps < 200:
            base_env.render(fog_of_war=fog_of_war)

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


def manual_control(env_id: str = "E7", reward_id: str = "simple", fog_of_war: bool = True):
    """Manual keyboard control for testing"""
    print("=" * 70)
    print("MANUAL CONTROL")
    print("=" * 70)
    print("\nControls:")
    print("  ↑   : Move up")
    print("  ↓   : Move down")
    print("  ←   : Move left")
    print("  →   : Move right")
    print("  R   : Reset")
    print("  F   : Toggle fog of war")
    print("  ESC : Quit")
    print("\nNote: Keys and doors auto-pickup/open when you move onto them")
    if fog_of_war:
        print("Fog of War: ON (only showing explored areas)")
    else:
        print("Fog of War: OFF (showing full map)")
    print("=" * 70)

    import pygame

    env = create_env(env_id, reward_id, "full_map", render=True)
    
    # Get base environment for rendering
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    obs, info = env.reset()

    print(f"\nAgent: {info['agent_pos']}, Goal: {info['goal_pos']}")
    print(f"Mission: {info['mission']}")

    running = True
    total_reward = 0
    steps = 0
    current_fog_of_war = fog_of_war

    while running:
        base_env.render(fog_of_war=current_fog_of_war)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                action = None

                if event.key == pygame.K_UP:
                    action = 0  # Move up
                elif event.key == pygame.K_DOWN:
                    action = 1  # Move down
                elif event.key == pygame.K_LEFT:
                    action = 2  # Move left
                elif event.key == pygame.K_RIGHT:
                    action = 3  # Move right
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    steps = 0
                    print(f"\n--- RESET ---")
                    print(f"Agent: {info['agent_pos']}, Goal: {info['goal_pos']}")
                elif event.key == pygame.K_f:
                    current_fog_of_war = not current_fog_of_war
                    print(f"Fog of War: {'ON (explored only)' if current_fog_of_war else 'OFF (full map)'}")
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
        default=None,
        choices=list(ALL_ENV_CONFIGS.keys()),
        help="Environment configuration (auto-detected from filename if not specified)",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default=None,
        choices=list(REWARD_WRAPPERS.keys()),
        help="Reward strategy (auto-detected from filename if not specified)",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default=None,
        choices=["full_map", "partial_view", "agent_view"],
        help="Observation mode (auto-detected from filename if not specified)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization (faster)")
    parser.add_argument("--delay", type=int, default=100, help="Delay between steps in ms")
    parser.add_argument("--full-map", action="store_true", help="Show full map instead of fog of war")
    parser.add_argument("--random", action="store_true", help="Run random agent instead of model")
    parser.add_argument("--manual", action="store_true", help="Manual keyboard control")

    args = parser.parse_args()

    if args.manual:
        manual_control(args.env or "E7", args.reward or "simple", fog_of_war=not args.full_map)
    elif args.random:
        run_random_agent(args.env or "E7", args.reward or "simple", args.episodes, args.delay, fog_of_war=not args.full_map)
    elif args.model_path:
        evaluate_model(
            model_path=args.model_path,
            env_id=args.env,
            reward_id=args.reward,
            obs_mode=args.obs,
            num_episodes=args.episodes,
            render=not args.no_render,
            delay_ms=args.delay,
            fog_of_war=not args.full_map,
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
