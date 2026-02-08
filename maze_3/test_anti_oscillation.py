"""
Test script to demonstrate anti-oscillation mechanisms.

This script shows how the new reward system prevents agents from
exploiting the reward by oscillating back and forth.
"""

import sys
sys.path.append('.')

from environments.base_env import LockedRoomEnv
from environments.reward_wrappers import SimpleRewardWrapper, SimpleObs

def test_oscillation():
    """Test that oscillation is properly penalized"""
    
    # Create environment with fixed positions for reproducibility
    env = LockedRoomEnv(
        size=19,
        observation_mode="full_map",
        render_mode=None,
        fixed_agent_pos=(9, 9),  # Start in corridor
        fixed_goal_pos=(3, 3),   # Goal in top-left room
        num_doors=0,  # No doors for simple test
        verbose=True
    )
    
    # Wrap with observation and reward wrappers
    env = SimpleObs(env)
    env = SimpleRewardWrapper(env)
    
    obs, info = env.reset()
    
    print("=" * 60)
    print("ANTI-OSCILLATION TEST")
    print("=" * 60)
    print(f"Agent starts at: {info['agent_pos']}")
    print(f"Goal is at: {info['goal_pos']}")
    print("\nTesting oscillation pattern (left-right-left-right)...\n")
    
    # Test oscillation: move left, right, left, right
    actions = [2, 3, 2, 3, 2, 3]  # LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT
    total_reward = 0
    
    for i, action in enumerate(actions):
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: {action_names[action]}")
        print(f"  Position: {info['agent_pos']}")
        print(f"  Reward breakdown:")
        for key, value in info.get('reward_breakdown', {}).items():
            if value != 0:
                print(f"    {key}: {value:.2f}")
        print(f"  Step reward: {reward:.2f}")
        print(f"  Total reward so far: {total_reward:.2f}")
        print()
        
        if terminated or truncated:
            break
    
    print("=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. First move LEFT gets distance reward (getting closer)")
    print("2. Moving RIGHT back gets NO distance reward (not beating best)")
    print("3. Revisiting positions incurs REVISIT PENALTY")
    print("4. Direction changes incur DIRECTION CHANGE PENALTY")
    print("5. Direct reversals get STRONGER penalties")
    print("\nTotal reward for oscillation pattern:", total_reward)
    print("This should be NEGATIVE, showing oscillation is punished!")
    print("=" * 60)


def test_progressive_movement():
    """Test that progressive movement toward goal is rewarded"""
    
    env = LockedRoomEnv(
        size=19,
        observation_mode="full_map",
        render_mode=None,
        fixed_agent_pos=(9, 9),  # Start in corridor
        fixed_goal_pos=(9, 3),   # Goal directly to the left
        num_doors=0,
        verbose=False
    )
    
    env = SimpleObs(env)
    env = SimpleRewardWrapper(env)
    
    obs, info = env.reset()
    
    print("\n" + "=" * 60)
    print("PROGRESSIVE MOVEMENT TEST")
    print("=" * 60)
    print(f"Agent starts at: {info['agent_pos']}")
    print(f"Goal is at: {info['goal_pos']}")
    print("\nMoving consistently toward goal (all LEFT)...\n")
    
    # Move consistently toward goal
    total_reward = 0
    for i in range(6):
        obs, reward, terminated, truncated, info = env.step(2)  # LEFT
        total_reward += reward
        
        print(f"Step {i+1}: Move LEFT")
        print(f"  Position: {info['agent_pos']}")
        print(f"  Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if terminated:
            print(f"\n🎉 GOAL REACHED! Final reward: {total_reward:.2f}")
            break
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. Each step toward goal gets distance reward")
    print("2. No revisit penalties (always moving to new positions)")
    print("3. No direction change penalties (consistent movement)")
    print("4. Progressive movement is REWARDED!")
    print("=" * 60)


if __name__ == "__main__":
    test_oscillation()
    test_progressive_movement()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The new anti-oscillation system includes:

1. **Best Distance Tracking**: Only rewards NEW minimum distances
   - Agent can't farm rewards by oscillating
   - Must make actual progress toward goal

2. **Revisit Penalties**: Penalizes returning to recent positions
   - Exponential penalty for repeated visits
   - Discourages back-and-forth movement

3. **Direction Change Penalties**: Penalizes changing direction
   - Mild penalty for any direction change
   - Strong penalty for direct reversals (180° turns)
   - Encourages momentum and consistent movement

4. **Position History**: Tracks last 10 positions
   - Detects oscillation patterns
   - Cleared when picking up keys or opening doors (new phase)

These mechanisms work together to prevent exploitation while still
rewarding genuine progress toward objectives.
""")
    print("=" * 60)
