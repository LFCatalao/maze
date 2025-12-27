"""
Test script to verify the no-doors configuration works correctly
"""

from locked_room_env import LockedRoomEnv

print("="*70)
print("TESTING: No Doors Configuration")
print("="*70)

# Create environment with no doors
env = LockedRoomEnv(
    observation_mode="full_map",
    task_type="full",
    render_mode="console",
    max_steps=500,
    num_doors=0,  # NO DOORS!
    randomize_doors=False,
    include_key=False,
    locked_door=False,
    include_mission_in_obs=True,
    verbose=True
)

print("\nEnvironment Configuration:")
print(f"  Number of doors: 0")
print(f"  Include key: False")
print(f"  Task: Reach the goal")

obs, info = env.reset()

print(f"\nMission: {info['mission']}")
print(f"Agent starting position: {info['agent_pos']}")
print(f"Carrying: {info['carrying']}")

print("\n" + "="*70)
print("INITIAL STATE - All rooms should be accessible via openings")
print("="*70)
env.render()

# Take a few random actions to verify movement works
print("\n" + "="*70)
print("Taking a few test actions...")
print("="*70)

actions = [2, 2, 2, 0, 2, 2]  # Forward, forward, forward, turn left, forward, forward
action_names = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]

for i, action in enumerate(actions):
    print(f"\nStep {i+1}: Action = {action_names[action]}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Position: {info['agent_pos']}, Reward: {reward}")
    
    if terminated:
        print("\n✓ GOAL REACHED!")
        break
    
    if truncated:
        print("\n✗ Episode truncated (max steps)")
        break

env.render()
env.close()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nVerify that:")
print("  ✓ All 6 door positions have openings (not walls)")
print("  ✓ Agent can move freely between rooms")
print("  ✓ Goal is reachable in one of the left-side rooms")
print("\nThis configuration is suitable for training basic navigation")
print("before introducing doors and keys.")
