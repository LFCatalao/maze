"""
Quick test to verify E7 environment configuration
"""
import sys
sys.path.append('maze_3')

from environments.configs import get_config
from environments.base_env import LockedRoomEnv
import numpy as np

def test_e7():
    """Test E7 environment with fixed key positions"""
    print("Testing E7 Configuration...")
    print("=" * 60)
    
    # Get E7 config
    config = get_config("E7")
    
    print(f"\nConfig: {config.name}")
    print(f"Description: {config.description}")
    print(f"Key features:")
    print(f"  - Fixed agent position: {config.fixed_agent_pos}")
    print(f"  - Random goal position: {config.fixed_goal_pos is None}")
    print(f"  - Number of doors: {config.num_doors}")
    print(f"  - Key chain enabled: {config.enable_key_chain}")
    print(f"  - Fixed key positions: {config.use_fixed_key_positions}")
    
    # Create environment
    env_kwargs = {
        'size': config.size,
        'observation_mode': config.observation_mode,
        'max_steps': config.max_steps,
        'agent_view_size': config.agent_view_size,
        'fixed_agent_pos': config.fixed_agent_pos,
        'fixed_goal_pos': config.fixed_goal_pos,
        'num_doors': config.num_doors,
        'randomize_doors': config.randomize_doors,
        'include_key': config.include_key,
        'locked_door': config.locked_door,
        'defined_doors': config.defined_doors,
        'goal_in_locked_room': config.goal_in_locked_room,
        'enable_key_chain': config.enable_key_chain,
        'use_fixed_key_positions': config.use_fixed_key_positions,
        'render_mode': 'console',
        'verbose': True,
    }
    
    env = LockedRoomEnv(**env_kwargs)
    
    print("\n" + "=" * 60)
    print("Testing multiple resets to verify key positions are fixed...")
    print("=" * 60)
    
    key_positions_per_reset = []
    
    for i in range(3):
        obs, info = env.reset(seed=i)
        print(f"\n--- Reset {i+1} (seed={i}) ---")
        print(f"Agent position: {info['agent_pos']}")
        print(f"Goal position: {info['goal_pos']}")
        print(f"Key positions: {env.key_positions}")
        print(f"Door positions: {list(env.door_positions.keys())}")
        
        # Store key positions for comparison
        key_positions_per_reset.append(sorted(env.key_positions.keys()))
    
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    
    # Check if key positions are in the canonical positions
    canonical_positions = {(3, 3), (9, 3), (15, 3), (3, 15), (9, 15), (15, 15)}
    
    all_keys_canonical = True
    for i, key_positions in enumerate(key_positions_per_reset):
        print(f"\nReset {i+1} key positions: {key_positions}")
        for pos in key_positions:
            if pos not in canonical_positions:
                print(f"  WARNING: Position {pos} is NOT a canonical position!")
                all_keys_canonical = False
            else:
                print(f"  ✓ Position {pos} is canonical")
    
    if all_keys_canonical:
        print("\n✓ SUCCESS: All keys are placed at canonical (fixed) positions!")
    else:
        print("\n✗ FAILURE: Some keys are not at canonical positions!")
    
    print("\nNote: The specific rooms and doors should vary between resets,")
    print("but the positions within each room should always be the same canonical positions.")
    
    env.close()

if __name__ == "__main__":
    test_e7()
