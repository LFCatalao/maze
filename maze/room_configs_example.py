"""
Example of different room configurations
Demonstrates progressive complexity levels
"""

from locked_room_env import LockedRoomEnv

print("="*60)
print("ROOM CONFIGURATION EXAMPLES")
print("="*60)

# Configuration 1: Single fixed door, no key
print("\n1. SIMPLEST: One fixed door, no key")
print("-"*60)
env1 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    render_mode="console",
    max_steps=100,
    num_doors=1,
    randomize_doors=False,
    include_key=False,
    locked_door=False
)
obs, info = env1.reset()
print(f"Mission: {info['mission']}")
print(f"Observation includes mission encoding: {'mission' in obs}")
if 'mission' in obs:
    mission_names = ["get_to_exit", "get_key", "open_door"]
    print(f"Mission encoded as: {obs['mission']} ({mission_names[obs['mission']]})")
env1.render()
env1.close()

# Configuration 2: Multiple fixed doors, no key
print("\n2. EASY: Three fixed doors, no key")
print("-"*60)
env2 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    render_mode="console",
    max_steps=100,
    num_doors=3,
    randomize_doors=False,
    include_key=False,
    locked_door=False
)
obs, info = env2.reset()
print(f"Mission: {info['mission']}")
env2.render()
env2.close()

# Configuration 3: Random doors, no key
print("\n3. MEDIUM: Three random doors, no key")
print("-"*60)
env3 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    render_mode="console",
    max_steps=100,
    num_doors=3,
    randomize_doors=True,
    include_key=False,
    locked_door=False
)
obs, info = env3.reset()
print(f"Mission: {info['mission']}")
env3.render()
env3.close()

# Configuration 4: One door with key (but not locked)
print("\n4. LEARNING KEYS: One door + key (unlocked)")
print("-"*60)
env4 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="get_key",
    render_mode="console",
    max_steps=100,
    num_doors=1,
    randomize_doors=False,
    include_key=True,
    locked_door=False
)
obs, info = env4.reset()
print(f"Mission: {info['mission']}")
env4.render()
env4.close()

# Configuration 5: One locked door with key
print("\n5. LOCKED DOOR: One locked door + key")
print("-"*60)
env5 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="unlock_door",
    render_mode="console",
    max_steps=100,
    num_doors=1,
    randomize_doors=False,
    include_key=True,
    locked_door=True
)
obs, info = env5.reset()
print(f"Mission: {info['mission']}")
env5.render()
env5.close()

# Configuration 6: Full complexity - multiple random doors + locked + key + goal
print("\n6. FULL TASK: All features")
print("-"*60)
env6 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="full",
    render_mode="console",
    max_steps=500,
    num_doors=6,
    randomize_doors=True,
    include_key=True,
    locked_door=True
)
obs, info = env6.reset()
print(f"Mission: {info['mission']}")
env6.render()
env6.close()

print("\n" + "="*60)
print("CURRICULUM LEARNING STRATEGY")
print("="*60)
print("""
Suggested training progression:

Stage 1: Single fixed door (config 1)
  - Learn basic navigation
  - Learn to reach doors
  
Stage 2: Multiple fixed doors (config 2)
  - Learn to find doors in different locations
  
Stage 3: Random doors (config 3)
  - Learn to generalize door positions
  
Stage 4: Keys without locks (config 4)
  - Learn to pick up objects
  - Learn object interaction
  
Stage 5: Locked doors (config 5)
  - Learn key-door matching
  - Learn unlocking sequence
  
Stage 6: Full task (config 6)
  - Combine all skills
  - Navigate to goal after unlocking
""")

print("\nTo use these configurations in training:")
print("  env = LockedRoomEnv(num_doors=1, include_key=False, locked_door=False)")
print("  # ... then wrap with TrainingLogger and FlattenObservation")
