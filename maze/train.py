"""
Training Script for Locked Room Environment

This script demonstrates how to train an RL agent on the environment
using stable-baselines3 (or any other RL library)

Requirements:
    pip install stable-baselines3 gymnasium pygame numpy tensorboard
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from locked_room_env import LockedRoomEnv
import torch
import os
from datetime import datetime


class FlattenObservation(gym.ObservationWrapper):
    """
    Wrapper to flatten the observation space for compatibility with
    standard RL algorithms
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate flattened observation space size
        image_size = np.prod(env.observation_space['image'].shape)
        
        # Flattened: image + direction (4) + carrying (7) + mission (3 if present)
        extra_size = 11  # direction + carrying
        if 'mission' in env.observation_space.spaces:
            extra_size += 3  # add mission encoding (0=get_to_exit, 1=get_key, 2=open_door)
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_size + extra_size,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Flatten the observation"""
        # Flatten image
        image_flat = obs['image'].flatten().astype(np.float32)
        
        # One-hot encode direction
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[obs['direction']] = 1.0
        
        # One-hot encode carrying
        carrying_onehot = np.zeros(7, dtype=np.float32)
        carrying_onehot[obs['carrying']] = 1.0
        
        # One-hot encode mission if present
        components = [image_flat, direction_onehot, carrying_onehot]
        if 'mission' in obs:
            mission_onehot = np.zeros(3, dtype=np.float32)
            mission_onehot[obs['mission']] = 1.0
            components.append(mission_onehot)
        
        # Concatenate
        flat_obs = np.concatenate(components)
        
        return flat_obs


class TrainingLogger(gym.Wrapper):
    """Wrapper to log episode statistics"""
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def reset(self, **kwargs):
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            success = self.current_episode_reward > 0
            self.episode_successes.append(success)
            
            # Print episode summary every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_successes = self.episode_successes[-10:]
                print(f"\nEpisode {len(self.episode_rewards)}:")
                print(f"  Avg Reward (last 10): {np.mean(recent_rewards):.3f}")
                print(f"  Success Rate (last 10): {np.mean(recent_successes)*100:.1f}%")
                print(f"  Avg Length (last 10): {np.mean(self.episode_lengths[-10:]):.1f}")
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_reward += reward
        self.current_episode_length += 1
        return obs, reward, terminated, truncated, info


def train_progressive():
    """
    Progressive training: train on easier tasks first, then harder ones
    """
    print("="*60)
    print("PROGRESSIVE TRAINING")
    print("="*60)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
    except ImportError:
        print("\nError: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3 tensorboard")
        return
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/progressive_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"\nLogs will be saved to: {log_dir}")
    print("To view training progress, run: tensorboard --logdir=logs")
    
    # Stage 1: Learn to reach doors (easiest)
    print("\n" + "-"*60)
    print("STAGE 1: Training to reach doors")
    print("-"*60)
    
    def make_env_stage1():
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type="reach_door",
            render_mode=None,
            max_steps=100
        )
        env = TrainingLogger(env)
        env = FlattenObservation(env)
        return env
    
    env1 = DummyVecEnv([make_env_stage1])
    
    # Create callbacks
    checkpoint_callback1 = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/stage1_checkpoints",
        name_prefix="reach_door"
    )
    
    model1 = PPO(
        "MlpPolicy",
        env1,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=f"{log_dir}/stage1"
    )
    
    print("Training for 50,000 steps...")
    print("Monitor progress: tensorboard --logdir=logs")
    model1.learn(total_timesteps=50_000, callback=checkpoint_callback1, progress_bar=True)
    model1.save(f"{log_dir}/stage1_reach_door")
    print(f"Stage 1 complete! Model saved to {log_dir}/stage1_reach_door")
    
    # Stage 2: Learn to get keys
    print("\n" + "-"*60)
    print("STAGE 2: Training to get keys")
    print("-"*60)
    
    def make_env_stage2():
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type="get_key",
            render_mode=None,
            max_steps=200
        )
        env = TrainingLogger(env)
        env = FlattenObservation(env)
        return env
    
    env2 = DummyVecEnv([make_env_stage2])
    
    checkpoint_callback2 = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/stage2_checkpoints",
        name_prefix="get_key"
    )
    
    # Can optionally load previous model and continue training
    model2 = PPO(
        "MlpPolicy",
        env2,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=f"{log_dir}/stage2"
    )
    
    print("Training for 100,000 steps...")
    model2.learn(total_timesteps=100_000, callback=checkpoint_callback2, progress_bar=True)
    model2.save(f"{log_dir}/stage2_get_key")
    print(f"Stage 2 complete! Model saved to {log_dir}/stage2_get_key")
    
    # Stage 3: Learn full task
    print("\n" + "-"*60)
    print("STAGE 3: Training full task")
    print("-"*60)
    
    def make_env_stage3():
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type="full",
            render_mode=None,
            max_steps=500
        )
        env = TrainingLogger(env)
        env = FlattenObservation(env)
        return env
    
    env3 = DummyVecEnv([make_env_stage3])
    
    checkpoint_callback3 = CheckpointCallback(
        save_freq=20000,
        save_path=f"{log_dir}/stage3_checkpoints",
        name_prefix="full_task"
    )
    
    model3 = PPO(
        "MlpPolicy",
        env3,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=f"{log_dir}/stage3"
    )
    
    print("Training for 200,000 steps...")
    model3.learn(total_timesteps=200_000, callback=checkpoint_callback3, progress_bar=True)
    model3.save(f"{log_dir}/stage3_full_task")
    print(f"Stage 3 complete! Model saved to {log_dir}/stage3_full_task")
    
    print("\n" + "="*60)
    print("PROGRESSIVE TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAll models and logs saved to: {log_dir}")
    print("\nTo view training curves:")
    print(f"  tensorboard --logdir={log_dir}")
    print("\nTo evaluate the final model:")
    print(f"  python train.py evaluate {log_dir}/stage3_full_task")


def train_single_task(
    task_type="full",
    observation_mode="full_map",
    total_timesteps=100_000,
    save_name=None
):
    """
    Train on a single task
    """
    print("="*60)
    print(f"TRAINING: {task_type.upper()} with {observation_mode.upper()}")
    print("="*60)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("\nError: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3 tensorboard")
        return
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_name is None:
        save_name = f"trained_{task_type}_{timestamp}"
    log_dir = f"logs/{save_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\nLogs will be saved to: {log_dir}")
    print("To view training progress, run: tensorboard --logdir=logs\n")
    
    # Create environment
    def make_env():
        env = LockedRoomEnv(
            observation_mode=observation_mode,
            task_type=task_type,
            render_mode=None,
            max_steps=500
        )
        env = TrainingLogger(env)
        env = FlattenObservation(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix=task_type
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=log_dir
    )
    
    # Train
    print(f"Training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)
    
    # Save
    final_path = f"{log_dir}/{save_name}"
    model.save(final_path)
    print(f"\nModel saved to: {final_path}.zip")
    print(f"\nTo view training curves: tensorboard --logdir={log_dir}")
    print(f"To evaluate: python train.py evaluate {final_path}")


def evaluate_model(model_path, num_episodes=10, render=True):
    """
    Evaluate a trained model
    """
    print("="*60)
    print(f"EVALUATING MODEL: {model_path}")
    print("="*60)
    
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("\nError: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    # Create environment
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="full",
        render_mode="human" if render else None,
        max_steps=500
    )
    env = FlattenObservation(env)
    
    # Run episodes
    successes = 0
    total_rewards = []
    total_steps = []
    keys_collected = 0
    doors_unlocked = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        collected_key = False
        unlocked_door = False
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Mission: {info['mission']}")
        print(f"{'='*50}")
        
        while not done:
            if render:
                env.render()
            
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step
            prev_carrying = info.get('carrying')
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Track key collection
            if not collected_key and info.get('carrying') is not None and prev_carrying is None:
                collected_key = True
                print("  ✓ Collected key!")
            
            # Simple way to detect door unlocking (reward increase in unlock_door task)
            if reward > 0 and not unlocked_door:
                unlocked_door = True
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if collected_key:
            keys_collected += 1
        if unlocked_door or episode_reward > 0:
            doors_unlocked += 1
        
        if episode_reward > 0:
            successes += 1
            print(f"\n  ✓✓✓ SUCCESS! ✓✓✓")
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Steps: {steps}")
        else:
            print(f"\n  ✗ Failed")
            print(f"  Steps: {steps}")
            if collected_key:
                print(f"  (Collected key but didn't complete)")
    
    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Keys Collected: {keys_collected}/{num_episodes} ({100*keys_collected/num_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    
    if successes > 0:
        successful_steps = [total_steps[i] for i in range(num_episodes) if total_rewards[i] > 0]
        print(f"Average Steps (successful): {np.mean(successful_steps):.1f}")
    
    env.close()


def custom_training_loop():
    """
    Example of a custom training loop without stable-baselines3
    This is useful if you want to implement your own RL algorithm
    """
    print("="*60)
    print("CUSTOM TRAINING LOOP EXAMPLE")
    print("="*60)
    
    # Create environment
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="reach_door",
        render_mode=None,
        max_steps=100
    )
    env = TrainingLogger(env)
    env = FlattenObservation(env)
    
    # Simple random policy (replace with your RL algorithm)
    num_episodes = 20
    
    print(f"\nRunning {num_episodes} episodes with random policy...")
    print("(Replace this with your own RL algorithm)\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 100:
            # Random action (replace with your policy)
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Here you would update your policy based on the transition
            # (obs, action, reward, next_obs, done)
    
    print("\nCustom training loop complete!")
    print("Episode statistics were logged by TrainingLogger wrapper")
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("TRAINING SCRIPT")
        print("="*60)
        print("\nUsage:")
        print("  python train.py progressive          - Progressive training")
        print("  python train.py single <task>        - Train single task")
        print("  python train.py evaluate <model>     - Evaluate a model")
        print("  python train.py custom               - Custom training loop")
        print("\nTasks: reach_door, get_key, unlock_door, full")
        print("\nExamples:")
        print("  python train.py progressive")
        print("  python train.py single full")
        print("  python train.py evaluate stage3_full_task")
        sys.exit(0)
    
    mode = sys.argv[1]
    
    if mode == "progressive":
        train_progressive()
    
    elif mode == "single":
        task = sys.argv[2] if len(sys.argv) > 2 else "full"
        train_single_task(
            task_type=task,
            observation_mode="full_map",
            total_timesteps=100_000,
            save_name=None
        )
    
    elif mode == "evaluate":
        if len(sys.argv) < 3:
            print("Error: Please specify model path")
            print("Example: python train.py evaluate stage3_full_task")
            sys.exit(1)
        
        model_path = sys.argv[2]
        evaluate_model(model_path, num_episodes=10, render=True)
    
    elif mode == "custom":
        custom_training_loop()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: progressive, single, evaluate, custom")