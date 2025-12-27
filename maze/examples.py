"""
Example Usage of the Locked Room Environment

This script demonstrates:
1. Different observation modes (full_map, partial_view, agent_view)
2. Different task types (reach_door, get_key, unlock_door, full)
3. Random agent behavior
4. Manual control
"""

import numpy as np
import pygame
from locked_room_env import LockedRoomEnv, Actions


def random_agent_demo(env, num_episodes=3, max_steps_per_episode=100):
    """Run a random agent to demonstrate the environment"""
    print(f"\n{'='*60}")
    print(f"Random Agent Demo")
    print(f"Observation Mode: {env.observation_mode}")
    print(f"Task Type: {env.task_type}")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"Mission: {info['mission']}")
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated) and steps < max_steps_per_episode:
            # Render
            env.render()
            
            # Random action
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Small delay for visualization
            if env.render_mode == "human":
                pygame.time.wait(100)
        
        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
    
    env.close()


def manual_control_demo(env):
    """Allow manual control of the agent"""
    print(f"\n{'='*60}")
    print(f"Manual Control Demo")
    print(f"Observation Mode: {env.observation_mode}")
    print(f"Task Type: {env.task_type}")
    print(f"{'='*60}")
    print("\nControls:")
    print("  Arrow Keys: Turn left/right, Move forward")
    print("  SPACE: Pick up")
    print("  E: Toggle (open door)")
    print("  R: Reset")
    print("  ESC: Quit")
    print("="*60 + "\n")
    
    obs, info = env.reset()
    print(f"Mission: {info['mission']}")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        env.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                action = None
                
                if event.key == pygame.K_LEFT:
                    action = Actions.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = Actions.RIGHT
                elif event.key == pygame.K_UP:
                    action = Actions.FORWARD
                elif event.key == pygame.K_SPACE:
                    action = Actions.PICKUP
                elif event.key == pygame.K_e:
                    action = Actions.TOGGLE
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print(f"\nReset! New mission: {info['mission']}")
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if reward > 0:
                        print(f"Success! Reward: {reward:.3f}")
                    
                    if terminated or truncated:
                        print(f"Episode finished! Steps: {info['step_count']}")
                        print("Press R to reset or ESC to quit")
        
        clock.tick(30)
    
    env.close()


def progressive_training_demo():
    """Demonstrate progressive task learning"""
    print(f"\n{'='*60}")
    print("Progressive Training Demo")
    print("This demonstrates how to train an agent progressively")
    print(f"{'='*60}\n")
    
    # Stage 1: Learn to reach doors
    print("STAGE 1: Learning to reach doors")
    print("-" * 40)
    env1 = LockedRoomEnv(
        observation_mode="full_map",
        task_type="reach_door",
        render_mode="console",
        max_steps=50
    )
    
    # Run a few episodes
    for episode in range(3):
        obs, info = env1.reset()
        print(f"\nEpisode {episode + 1}: {info['mission']}")
        
        for step in range(20):
            action = env1.action_space.sample()
            obs, reward, terminated, truncated, info = env1.step(action)
            
            if terminated:
                print(f"Success at step {step + 1}! Reward: {reward:.3f}")
                break
    
    env1.close()
    
    # Stage 2: Learn to get keys
    print("\n\nSTAGE 2: Learning to get keys")
    print("-" * 40)
    env2 = LockedRoomEnv(
        observation_mode="full_map",
        task_type="get_key",
        render_mode="console",
        max_steps=100
    )
    
    for episode in range(2):
        obs, info = env2.reset()
        print(f"\nEpisode {episode + 1}: {info['mission']}")
        
        for step in range(30):
            action = env2.action_space.sample()
            obs, reward, terminated, truncated, info = env2.step(action)
            
            if terminated:
                print(f"Success at step {step + 1}! Reward: {reward:.3f}")
                break
    
    env2.close()
    
    # Stage 3: Learn to unlock doors
    print("\n\nSTAGE 3: Learning to unlock doors")
    print("-" * 40)
    env3 = LockedRoomEnv(
        observation_mode="partial_view",
        task_type="unlock_door",
        render_mode="console",
        max_steps=200
    )
    
    for episode in range(2):
        obs, info = env3.reset()
        print(f"\nEpisode {episode + 1}: {info['mission']}")
        
        for step in range(40):
            action = env3.action_space.sample()
            obs, reward, terminated, truncated, info = env3.step(action)
            
            if terminated:
                print(f"Success at step {step + 1}! Reward: {reward:.3f}")
                break
    
    env3.close()
    
    print("\n\nProgressive training complete!")
    print("The agent would learn each stage before moving to the next.")


def compare_observation_modes():
    """Compare different observation modes"""
    print(f"\n{'='*60}")
    print("Observation Modes Comparison")
    print(f"{'='*60}\n")
    
    modes = ["full_map", "partial_view", "agent_view"]
    
    for mode in modes:
        print(f"\n{mode.upper()}")
        print("-" * 40)
        
        env = LockedRoomEnv(
            observation_mode=mode,
            task_type="full",
            render_mode="console",
            max_steps=50
        )
        
        obs, info = env.reset()
        print(f"Observation shape: {obs['image'].shape}")
        print(f"Mission: {info['mission']}")
        print(f"Agent can see: ", end="")
        
        if mode == "full_map":
            print("entire grid (omniscient view)")
        elif mode == "partial_view":
            print("local area around agent")
        else:
            print("first-person view (what's in front)")
        
        env.close()


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("LOCKED ROOM ENVIRONMENT - EXAMPLES")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nAvailable demos:")
        print("1. random   - Random agent with pygame visualization")
        print("2. manual   - Manual control with keyboard")
        print("3. progress - Progressive training demonstration")
        print("4. compare  - Compare observation modes")
        print("\nUsage: python examples.py [mode]")
        print("Example: python examples.py manual")
        
        mode = input("\nSelect demo (random/manual/progress/compare): ").strip().lower()
    
    if mode == "random":
        # Random agent with full map observation
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type="full",
            render_mode="human",
            max_steps=200
        )
        random_agent_demo(env, num_episodes=3, max_steps_per_episode=200)
    
    elif mode == "manual":
        # Manual control
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type="full",
            render_mode="human",
            max_steps=500
        )
        manual_control_demo(env)
    
    elif mode == "progress":
        # Progressive training
        progressive_training_demo()
    
    elif mode == "compare":
        # Compare observation modes
        compare_observation_modes()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: random, manual, progress, compare")
