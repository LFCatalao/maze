"""
PPO Training Script for Locked Room Environment

Simple script to train PPO on the environment with customizable checkpointing.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from locked_room_env import LockedRoomEnv
import os
from datetime import datetime


class FlattenObservation(gym.ObservationWrapper):
    """Flatten the observation space for PPO"""
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate flattened observation space size
        image_size = np.prod(env.observation_space['image'].shape)
        
        # Flattened: image + direction (4) + carrying (7) + mission (3 if present)
        extra_size = 11  # direction + carrying
        if 'mission' in env.observation_space.spaces:
            extra_size += 3  # add mission encoding
        
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


class CustomCheckpointCallback:
    """Custom callback to save checkpoints with specific naming format"""
    def __init__(self, save_freq, save_path, model_name, task_name):
        """
        Args:
            save_freq: Save every N steps
            save_path: Directory to save checkpoints
            model_name: Name of the model (e.g., "ppo")
            task_name: Task being trained (e.g., "reach_goal_no_doors")
        """
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name
        self.task_name = task_name
        self.n_calls = 0
        
        os.makedirs(save_path, exist_ok=True)
    
    def __call__(self, locals_dict, globals_dict):
        """Called at each step"""
        self.n_calls += 1
        
        # Save checkpoint
        if self.n_calls % self.save_freq == 0:
            model = locals_dict['self']
            checkpoint_path = os.path.join(
                self.save_path,
                f"{self.model_name}_{self.task_name}_{self.n_calls}"
            )
            model.save(checkpoint_path)
            print(f"\n[Checkpoint] Saved model to {checkpoint_path}.zip")
        
        return True  # Continue training


def train_ppo_no_doors(
    total_timesteps=500_000,
    checkpoint_freq=50_000,
    model_name="ppo",
    task_name="reach_goal_no_doors",
    save_dir="models",
    log_dir="logs"
):
    """
    Train PPO to reach the goal without any doors
    
    Args:
        total_timesteps: Total training steps
        checkpoint_freq: Save checkpoint every N steps
        model_name: Name for the model
        task_name: Task description for naming
        save_dir: Directory to save model checkpoints
        log_dir: Directory for tensorboard logs
    """
    print("="*70)
    print("PPO TRAINING - REACH GOAL (NO DOORS)")
    print("="*70)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("\nError: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        return
    
    # Check if tensorboard is available
    try:
        import tensorboard
        tensorboard_available = True
    except ImportError:
        tensorboard_available = False
        print("\nWarning: tensorboard not installed - logging will be disabled")
        print("To enable tensorboard logging: pip install tensorboard\n")
    
    # Check if progress bar is available
    try:
        import tqdm
        import rich
        progress_bar = True
    except ImportError:
        progress_bar = False
        print("Warning: tqdm/rich not installed - progress bar disabled")
        print("To enable progress bar: pip install tqdm rich\n")
    
    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{model_name}_{task_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    if tensorboard_available:
        tb_log_path = os.path.join(log_dir, f"{model_name}_{task_name}_{timestamp}")
        os.makedirs(tb_log_path, exist_ok=True)
    else:
        tb_log_path = None
    
    print(f"\nConfiguration:")
    print(f"  Task: Reach goal without doors")
    print(f"  Observation: Full map")
    print(f"  Reward: 1.0 only at goal")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Checkpoint frequency: {checkpoint_freq:,} steps")
    print(f"\nSave locations:")
    print(f"  Checkpoints: {save_path}")
    if tensorboard_available:
        print(f"  TensorBoard logs: {tb_log_path}")
        print(f"\nTo monitor training:")
        print(f"  tensorboard --logdir={log_dir}")
    else:
        print(f"  TensorBoard: Disabled (not installed)")
    print("="*70)
    
    # Create environment - NO DOORS, just reach the goal
    def make_env():
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type="full",  # Just reach the goal
            render_mode=None,
            max_steps=500,
            num_doors=0,  # NO DOORS!
            randomize_doors=False,
            include_key=False,
            locked_door=False,
            include_mission_in_obs=True,
            verbose=False  # Disable prints during training
        )
        env = FlattenObservation(env)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    print("\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Create custom callback for checkpointing
    class CheckpointCallback(BaseCallback):
        def __init__(self, save_freq, save_path, model_name, task_name, verbose=0):
            super().__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path
            self.model_name = model_name
            self.task_name = task_name
        
        def _on_step(self):
            if self.n_calls % self.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.save_path,
                    f"{self.model_name}_{self.task_name}_{self.n_calls}"
                )
                self.model.save(checkpoint_path)
                if self.verbose > 0:
                    print(f"\n[Checkpoint] Saved at step {self.n_calls}: {checkpoint_path}.zip")
            return True
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        model_name=model_name,
        task_name=task_name,
        verbose=1
    )
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.5,  # MUCH higher entropy to force exploration
        tensorboard_log=tb_log_path
    )
    
    print("\nPPO Hyperparameters:")
    print(f"  Learning rate: 3e-4")
    print(f"  Steps per update: 2048")
    print(f"  Batch size: 64")
    print(f"  Epochs: 10")
    print(f"  Gamma: 0.99")
    print(f"  GAE Lambda: 0.95")
    print(f"  Clip range: 0.2")
    print(f"  Entropy coefficient: 0.1 (encourages exploration)")
    print(f"  Entropy coefficient: 0.01")
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Training for {total_timesteps:,} steps...")
    print("Press Ctrl+C to stop training early\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=progress_bar
        )
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_path = os.path.join(save_path, f"{model_name}_{task_name}_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total steps completed: {model.num_timesteps:,}")
    print(f"Checkpoints saved in: {save_path}")
    if tensorboard_available:
        print(f"TensorBoard logs: {tb_log_path}")
        print("\nTo view training curves:")
        print(f"  tensorboard --logdir={log_dir}")
    print("\nTo evaluate the model:")
    print(f"  python evaluate_ppo.py {final_path}")
    print("="*70)
    
    env.close()
    return model, final_path


if __name__ == "__main__":
    import sys
    
    # Default parameters
    total_timesteps = 500_000
    checkpoint_freq = 50_000
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        total_timesteps = int(sys.argv[1])
    if len(sys.argv) > 2:
        checkpoint_freq = int(sys.argv[2])
    
    print("\n" + "="*70)
    print("PPO TRAINING SCRIPT")
    print("="*70)
    print("\nUsage:")
    print("  python train_ppo.py [total_steps] [checkpoint_freq]")
    print("\nExamples:")
    print("  python train_ppo.py                    # Default: 500k steps, checkpoint every 50k")
    print("  python train_ppo.py 1000000            # 1M steps, checkpoint every 50k")
    print("  python train_ppo.py 1000000 100000     # 1M steps, checkpoint every 100k")
    print("="*70 + "\n")
    
    # Train the model
    train_ppo_no_doors(
        total_timesteps=total_timesteps,
        checkpoint_freq=checkpoint_freq,
        model_name="ppo",
        task_name="reach_goal_no_doors",
        save_dir="models",
        log_dir="logs"
    )
