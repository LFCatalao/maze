"""
Evaluation Script for Trained PPO Models
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from locked_room_env import LockedRoomEnv
import sys
import os


class FlattenObservation(gym.ObservationWrapper):
    """Flatten the observation space"""
    def __init__(self, env):
        super().__init__(env)
        
        image_size = np.prod(env.observation_space['image'].shape)
        extra_size = 11  # direction + carrying
        if 'mission' in env.observation_space.spaces:
            extra_size += 3
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size + extra_size,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        image_flat = obs['image'].flatten().astype(np.float32)
        
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[obs['direction']] = 1.0
        
        carrying_onehot = np.zeros(7, dtype=np.float32)
        carrying_onehot[obs['carrying']] = 1.0
        
        components = [image_flat, direction_onehot, carrying_onehot]
        if 'mission' in obs:
            mission_onehot = np.zeros(3, dtype=np.float32)
            mission_onehot[obs['mission']] = 1.0
            components.append(mission_onehot)
        
        return np.concatenate(components)


def evaluate_model(model_path, num_episodes=10, render=True, max_steps=500):
    """
    Evaluate a trained PPO model
    
    Args:
        model_path: Path to saved model (without .zip)
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        max_steps: Maximum steps per episode
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Error: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        return
    
    print("="*70)
    print(f"EVALUATING MODEL: {model_path}")
    print("="*70)
    
    # Load model
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\nError: Model file not found: {model_path}.zip")
        return
    
    print(f"\nLoading model...")
    model = PPO.load(model_path)
    print("Model loaded successfully!\n")
    
    # Create environment (matching training config)
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="full",
        render_mode="human" if render else None,
        max_steps=max_steps,
        num_doors=0,  # NO DOORS - same as training
        randomize_doors=False,
        include_key=False,
        locked_door=False,
        include_mission_in_obs=True
    )
    env = FlattenObservation(env)
    
    print(f"\nEnvironment Info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action space n: {env.action_space.n}")
    
    # Get initial observation to check shape
    test_obs, _ = env.reset()
    print(f"  Actual observation shape: {test_obs.shape}")
    print(f"  Observation dtype: {test_obs.dtype}")
    
    # Run evaluation
    successes = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Mission: {info['mission']}")
        print(f"Agent starting position: {info['agent_pos']}")
        print(f"{'='*50}")
        
        # Action names for logging
        action_names = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "PICKUP", 4: "DROP", 5: "TOGGLE", 6: "DONE"}
        
        while not done:
            if render:
                import pygame
                # Render first (this initializes pygame if needed)
                env.render()
                
                # Process pygame events to keep window responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            
            # Store previous position
            prev_pos = info['agent_pos'].copy() if 'agent_pos' in info else None
            
            # Predict action (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # Debug action type and value
            if steps < 5:  # Only log first few steps in detail
                print(f"\n  [DEBUG] action type: {type(action)}, value: {action}, shape: {getattr(action, 'shape', 'N/A')}")
            
            # Convert action to int - handle numpy scalar
            action_int = int(action)
            
            # Log action
            print(f"Step {steps+1}: Action={action_names.get(action_int, action_int)} ({action_int}), Pos={prev_pos}", end="")
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Log result
            new_pos = info['agent_pos']
            if prev_pos != new_pos:
                print(f" -> {new_pos} [MOVED]", end="")
            else:
                print(f" [NO MOVE]", end="")
            if reward != 0:
                print(f" Reward={reward:.2f}", end="")
            print()  # New line
            
            # Optional: Add delay for visualization
            if render:
                pygame.time.wait(100)  # Increased to 100ms for better visibility
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if episode_reward > 0:
            successes += 1
            print(f"\n  ✓✓✓ SUCCESS! ✓✓✓")
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Steps: {steps}")
        else:
            print(f"\n  ✗ Failed")
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Steps: {steps}")
    
    # Print statistics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(total_rewards):.3f}")
    print(f"  Std:  {np.std(total_rewards):.3f}")
    print(f"  Min:  {np.min(total_rewards):.3f}")
    print(f"  Max:  {np.max(total_rewards):.3f}")
    print(f"\nStep Statistics:")
    print(f"  Mean: {np.mean(total_steps):.1f}")
    print(f"  Std:  {np.std(total_steps):.1f}")
    print(f"  Min:  {np.min(total_steps)}")
    print(f"  Max:  {np.max(total_steps)}")
    
    if successes > 0:
        successful_steps = [total_steps[i] for i in range(num_episodes) if total_rewards[i] > 0]
        print(f"\nSuccessful Episodes:")
        print(f"  Mean steps: {np.mean(successful_steps):.1f}")
        print(f"  Min steps:  {np.min(successful_steps)}")
        print(f"  Max steps:  {np.max(successful_steps)}")
    
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("PPO MODEL EVALUATION")
        print("="*70)
        print("\nUsage:")
        print("  python evaluate_ppo.py <model_path> [num_episodes] [render]")
        print("\nArguments:")
        print("  model_path    : Path to model file (without .zip extension)")
        print("  num_episodes  : Number of episodes to run (default: 10)")
        print("  render        : 'true' or 'false' (default: true)")
        print("\nExamples:")
        print("  python evaluate_ppo.py models/ppo_reach_goal_no_doors_final")
        print("  python evaluate_ppo.py models/ppo_reach_goal_no_doors_500000 20")
        print("  python evaluate_ppo.py models/ppo_reach_goal_no_doors_final 50 false")
        print("="*70)
        sys.exit(0)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    render = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
    
    evaluate_model(model_path, num_episodes, render)
