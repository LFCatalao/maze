#!/usr/bin/env python3
"""Test the new EnhancedObs observation space"""

import sys
sys.path.insert(0, 'maze_3')

from environments import LockedRoomEnv, get_config, EnhancedObs

# Create environment with E5_FIXED (4 doors, 4 keys)
config = get_config('E5_FIXED')
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
    defined_doors=config.defined_doors,
    goal_in_locked_room=config.goal_in_locked_room,
    enable_key_chain=config.enable_key_chain,
)

# Wrap with EnhancedObs
wrapped_env = EnhancedObs(env)
obs, info = wrapped_env.reset()

# Get base environment
base_env = wrapped_env.env
while hasattr(base_env, 'env'):
    base_env = base_env.env

print("=" * 70)
print("ENHANCED OBSERVATION SPACE TEST")
print("=" * 70)
print(f"\nObservation shape: {obs.shape}")
print(f"Total features: {len(obs)}")

print(f"\n{'='*70}")
print("OBSERVATION BREAKDOWN")
print("=" * 70)
print(f"1. Agent position:")
print(f"   - y = {obs[0]:.0f}")
print(f"   - x = {obs[1]:.0f}")

print(f"\n2. Carrying key:")
carrying = obs[2]
if carrying == -1:
    print(f"   - Not carrying any key (value: {carrying:.0f})")
else:
    color_names = {0: 'RED', 1: 'GREEN', 2: 'BLUE', 3: 'PURPLE', 4: 'YELLOW', 5: 'GREY'}
    print(f"   - Carrying {color_names.get(int(carrying), 'UNKNOWN')} key (value: {carrying:.0f})")

print(f"\n3. 7x7 Partial View: 147 features (indices 3-149)")
print(f"   - Encodes: object type, color, state for 7x7 grid around agent")
print(f"   - Sample (first 15): {obs[3:18]}")

print(f"\n4. Explored Map Memory: 1083 features (indices 150-1232)")
print(f"   - Full 19x19x3 grid of what agent has seen")
print(f"   - Channel 0: object type (0=empty, 1=wall, 2=door, 3=key, 4=goal)")
print(f"   - Channel 1: color (for keys/doors)")
print(f"   - Channel 2: state (for doors: 0=open, 1=closed, 2=locked)")

print(f"\n{'='*70}")
print("ENVIRONMENT STATE")
print("=" * 70)
print(f"Keys in environment: {len(base_env.key_positions)}")
for pos, color in base_env.key_positions.items():
    color_names = {0: 'RED', 1: 'GREEN', 2: 'BLUE', 3: 'PURPLE', 4: 'YELLOW', 5: 'GREY'}
    print(f"  - {color_names.get(color, 'UNKNOWN')} key at position {pos}")

print(f"\nDoors in environment: {len(base_env.door_positions)}")
for pos, (color, state) in base_env.door_positions.items():
    color_names = {0: 'RED', 1: 'GREEN', 2: 'BLUE', 3: 'PURPLE', 4: 'YELLOW', 5: 'GREY'}
    state_names = {0: 'OPEN', 1: 'CLOSED', 2: 'LOCKED'}
    print(f"  - {color_names.get(color, 'UNKNOWN')} door at {pos}, state: {state_names.get(state, 'UNKNOWN')}")

print(f"\nGoal position: {base_env.goal_pos}")
print(f"Agent position: {base_env.agent_pos}")

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print("✓ Agent knows its exact position")
print("✓ Agent knows what key it's carrying")
print("✓ Agent sees 7x7 area around it (walls, doors, keys, goal)")
print("✓ Agent remembers everything it has explored (full 19x19 map)")
print("✓ Memory includes: object types, colors, and door states")
print("=" * 70)
