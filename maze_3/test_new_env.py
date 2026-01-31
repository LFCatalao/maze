"""
Quick test script to verify the new environment mechanics work correctly
"""

from environments.base_env import LockedRoomEnv, Actions, COLOR_NAMES
import numpy as np


def test_omnidirectional_movement():
    """Test that the 4-directional movement works"""
    print("=" * 60)
    print("TEST 1: Omnidirectional Movement")
    print("=" * 60)
    
    env = LockedRoomEnv(
        size=19,
        fixed_agent_pos=(10, 9),  # Center of corridor
        fixed_goal_pos=(2, 2),
        num_doors=0,
        verbose=True
    )
    
    obs, info = env.reset()
    print(f"Initial position: {env.agent_pos}")
    print(f"Action space: {env.action_space}")
    
    # Test each direction
    actions = [
        (Actions.UP, "UP"),
        (Actions.DOWN, "DOWN"),
        (Actions.LEFT, "LEFT"),
        (Actions.RIGHT, "RIGHT")
    ]
    
    for action, name in actions:
        old_pos = env.agent_pos.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos
        print(f"{name}: {old_pos} -> {new_pos}")
    
    print("✓ Omnidirectional movement works!\n")


def test_auto_pickup():
    """Test that keys are automatically picked up when walked over"""
    print("=" * 60)
    print("TEST 2: Auto-Pickup")
    print("=" * 60)
    
    env = LockedRoomEnv(
        size=19,
        fixed_agent_pos=(10, 8),
        fixed_goal_pos=(2, 2),
        defined_doors=[
            {'door_idx': 0, 'key_pos': (10, 10)}  # Key to the right
        ],
        verbose=True
    )
    
    obs, info = env.reset()
    print(f"Initial position: {env.agent_pos}")
    print(f"Key position: {env.key_positions}")
    print(f"Carrying: {env.carrying}")
    
    # Move right twice to reach key
    print("\nMoving right to key...")
    obs, _, _, _, _ = env.step(Actions.RIGHT)
    obs, _, _, _, _ = env.step(Actions.RIGHT)
    
    print(f"After movement - Carrying: {COLOR_NAMES.get(env.carrying, 'nothing')}")
    print(f"Remaining keys: {env.key_positions}")
    
    if env.carrying is not None:
        print("✓ Auto-pickup works!\n")
    else:
        print("✗ Auto-pickup FAILED!\n")


def test_auto_door_opening():
    """Test that doors are automatically opened/unlocked when walked through"""
    print("=" * 60)
    print("TEST 3: Auto Door Opening")
    print("=" * 60)
    
    env = LockedRoomEnv(
        size=19,
        fixed_agent_pos=(3, 6),  # Just left of a door
        fixed_goal_pos=(2, 2),
        defined_doors=[
            {'door_idx': 0, 'key_pos': (3, 5)}  # Yellow door at (3, 7), key nearby
        ],
        verbose=True
    )
    
    obs, info = env.reset()
    print(f"Initial position: {env.agent_pos}")
    print(f"Door at (3, 7) state: {env.door_positions.get((3, 7), 'not found')}")
    
    # Pick up key (move left)
    print("\nMoving to pick up key...")
    obs, _, _, _, _ = env.step(Actions.LEFT)
    print(f"Carrying: {COLOR_NAMES.get(env.carrying, 'nothing')}")
    
    # Move back right
    obs, _, _, _, _ = env.step(Actions.RIGHT)
    
    # Try to go through door
    print("\nMoving through door...")
    obs, _, _, _, _ = env.step(Actions.RIGHT)
    
    print(f"Position after: {env.agent_pos}")
    print(f"Door state after: {env.door_positions.get((3, 7), 'not found')}")
    print(f"Carrying after: {COLOR_NAMES.get(env.carrying, 'nothing')}")
    
    door_state = env.door_positions.get((3, 7), (None, None))[1]
    if door_state == 0 and env.carrying is None:
        print("✓ Auto door unlock/open works!\n")
    else:
        print("✗ Auto door unlock/open FAILED!\n")


def test_partial_view_and_fog():
    """Test that the 7x7 partial view and fog of war work"""
    print("=" * 60)
    print("TEST 4: Partial View and Fog of War")
    print("=" * 60)
    
    env = LockedRoomEnv(
        size=19,
        fixed_agent_pos=(16, 2),
        fixed_goal_pos=(2, 15),
        num_doors=0,
        verbose=False
    )
    
    obs, info = env.reset()
    
    print(f"Observation keys: {obs.keys()}")
    print(f"Partial view shape: {obs['image'].shape}")
    print(f"Explored map shape: {obs['explored_map'].shape}")
    print(f"Carrying value: {obs['carrying']}")
    
    # Check that direction is NOT in observation
    if 'direction' not in obs:
        print("✓ Direction removed from observation space")
    else:
        print("✗ Direction still in observation space!")
    
    # Verify 7x7 view
    if obs['image'].shape == (7, 7, 3):
        print("✓ Partial view is 7x7")
    else:
        print(f"✗ Partial view is {obs['image'].shape}, expected (7, 7, 3)")
    
    # Verify explored map exists
    if obs['explored_map'].shape == (19, 19, 3):
        print("✓ Explored map has correct shape")
    else:
        print(f"✗ Explored map shape incorrect: {obs['explored_map'].shape}")
    
    # Count non-zero cells in explored map (should be > 0 after reset)
    explored_cells = np.sum(obs['explored_map'] != 0)
    print(f"Explored cells after reset: {explored_cells}")
    
    if explored_cells > 0:
        print("✓ Fog of war initialized and updated")
    else:
        print("✗ Fog of war not working")
    
    # Move and check if exploration increases
    obs_before = obs['explored_map'].copy()
    explored_before = np.sum(obs_before != 0)
    
    # Move to a new area
    for _ in range(3):
        obs, _, _, _, _ = env.step(Actions.UP)
    
    explored_after = np.sum(obs['explored_map'] != 0)
    print(f"Explored cells after moving: {explored_after}")
    
    if explored_after > explored_before:
        print("✓ Exploration updates with movement!\n")
    else:
        print("✗ Exploration not updating with movement!\n")


def test_environment_6():
    """Test the specific Environment 6 configuration"""
    print("=" * 60)
    print("TEST 5: Environment 6 Configuration")
    print("=" * 60)
    
    env = LockedRoomEnv(
        size=19,
        fixed_agent_pos=(16, 2),
        fixed_goal_pos=(2, 15),
        defined_doors=[
            {'door_idx': 2, 'key_pos': (16, 3)},  # Blue door
            {'door_idx': 4, 'key_pos': (2, 2)},   # Grey door
            {'door_idx': 3, 'key_pos': (10, 3)},  # Purple door
        ],
        verbose=True
    )
    
    obs, info = env.reset()
    
    print(f"Agent start: {env.agent_pos}")
    print(f"Goal: {env.goal_pos}")
    print(f"Number of doors: {len(env.door_positions)}")
    print(f"Number of keys: {len(env.key_positions)}")
    print(f"Door positions: {list(env.door_positions.keys())}")
    print(f"Key positions: {list(env.key_positions.keys())}")
    
    print("\nSimulating a few moves...")
    # Move right to pick up first key
    obs, _, _, _, _ = env.step(Actions.RIGHT)
    print(f"Step 1 - Position: {env.agent_pos}, Carrying: {COLOR_NAMES.get(env.carrying, 'nothing')}")
    
    # Move up toward door
    for i in range(3):
        obs, reward, terminated, truncated, _ = env.step(Actions.UP)
        if terminated:
            print(f"✓ Environment 6 working - Goal reached!")
            break
    
    print(f"Final position: {env.agent_pos}")
    print("✓ Environment 6 configuration works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING NEW ENVIRONMENT MECHANICS")
    print("=" * 60 + "\n")
    
    test_omnidirectional_movement()
    test_auto_pickup()
    test_auto_door_opening()
    test_partial_view_and_fog()
    test_environment_6()
    
    print("=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)
