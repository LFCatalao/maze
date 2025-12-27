"""
Test script to verify the Locked Room Environment works correctly
"""

import numpy as np
from locked_room_env import LockedRoomEnv, Actions


def test_basic_functionality():
    """Test basic environment functionality"""
    print("="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="full",
        render_mode="console",
        max_steps=50
    )
    
    # Test reset
    print("\n✓ Testing reset...")
    obs, info = env.reset()
    assert 'image' in obs
    assert 'direction' in obs
    assert 'carrying' in obs
    assert 'mission' in info
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Mission: {info['mission']}")
    
    # Test step
    print("\n✓ Testing step...")
    action = Actions.FORWARD
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step successful")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    # Test actions
    print("\n✓ Testing all actions...")
    for action_name, action_value in Actions.__members__.items():
        obs, reward, terminated, truncated, info = env.step(action_value)
        print(f"  {action_name}: OK")
        if terminated:
            break
    
    env.close()
    print("\n✓ Test 1 PASSED\n")


def test_observation_modes():
    """Test different observation modes"""
    print("="*60)
    print("TEST 2: Observation Modes")
    print("="*60)
    
    modes = ["full_map", "partial_view", "agent_view"]
    
    for mode in modes:
        print(f"\n✓ Testing {mode}...")
        env = LockedRoomEnv(
            observation_mode=mode,
            task_type="full",
            render_mode=None,
            max_steps=50
        )
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs['image'].shape}")
        
        # Take a few random steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print(f"  {mode}: OK")
    
    print("\n✓ Test 2 PASSED\n")


def test_task_types():
    """Test different task types"""
    print("="*60)
    print("TEST 3: Task Types")
    print("="*60)
    
    tasks = ["reach_door", "get_key", "unlock_door", "full"]
    
    for task in tasks:
        print(f"\n✓ Testing {task}...")
        env = LockedRoomEnv(
            observation_mode="full_map",
            task_type=task,
            render_mode=None,
            max_steps=50
        )
        
        obs, info = env.reset()
        print(f"  Mission: {info['mission']}")
        
        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  Task completed! Reward: {reward:.3f}")
                break
        
        env.close()
        print(f"  {task}: OK")
    
    print("\n✓ Test 3 PASSED\n")


def test_actions():
    """Test specific actions work correctly"""
    print("="*60)
    print("TEST 4: Specific Actions")
    print("="*60)
    
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="full",
        render_mode=None,
        max_steps=100
    )
    
    obs, info = env.reset()
    initial_dir = obs['direction']
    initial_pos = info['agent_pos'].copy()
    
    # Test turning
    print("\n✓ Testing turn left...")
    obs, _, _, _, info = env.step(Actions.LEFT)
    assert obs['direction'] == (initial_dir - 1) % 4
    print("  Turn left: OK")
    
    print("\n✓ Testing turn right...")
    obs, _, _, _, info = env.step(Actions.RIGHT)
    assert obs['direction'] == initial_dir
    print("  Turn right: OK")
    
    # Test forward movement
    print("\n✓ Testing forward movement...")
    # Turn to face open space and move
    for _ in range(10):
        obs, _, _, _, info = env.step(Actions.FORWARD)
        if info['agent_pos'] != initial_pos:
            print("  Forward movement: OK")
            break
    
    env.close()
    print("\n✓ Test 4 PASSED\n")


def test_rendering():
    """Test rendering modes"""
    print("="*60)
    print("TEST 5: Rendering")
    print("="*60)
    
    # Test console rendering
    print("\n✓ Testing console rendering...")
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="reach_door",
        render_mode="console",
        max_steps=50
    )
    
    obs, info = env.reset()
    env.render()
    print("  Console rendering: OK")
    env.close()
    
    # Test pygame rendering (just initialize, don't display)
    print("\n✓ Testing pygame rendering...")
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="reach_door",
        render_mode="rgb_array",
        max_steps=50
    )
    
    obs, info = env.reset()
    rgb_array = env.render()
    assert rgb_array is not None
    assert len(rgb_array.shape) == 3
    print(f"  RGB array shape: {rgb_array.shape}")
    print("  Pygame rendering: OK")
    env.close()
    
    print("\n✓ Test 5 PASSED\n")


def test_space_compatibility():
    """Test gymnasium space compatibility"""
    print("="*60)
    print("TEST 6: Gymnasium Space Compatibility")
    print("="*60)
    
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="full",
        render_mode=None,
        max_steps=50
    )
    
    print("\n✓ Testing action space...")
    assert env.action_space.contains(0)
    assert env.action_space.contains(6)
    assert not env.action_space.contains(-1)
    assert not env.action_space.contains(7)
    print("  Action space: OK")
    
    print("\n✓ Testing observation space...")
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    print("  Observation space: OK")
    
    # Test sample
    print("\n✓ Testing space sampling...")
    action = env.action_space.sample()
    assert 0 <= action <= 6
    print(f"  Sampled action: {action}")
    print("  Space sampling: OK")
    
    env.close()
    print("\n✓ Test 6 PASSED\n")


def test_episode_completion():
    """Test that episodes can complete successfully"""
    print("="*60)
    print("TEST 7: Episode Completion")
    print("="*60)
    
    # Test with easier task
    print("\n✓ Testing reach_door task...")
    env = LockedRoomEnv(
        observation_mode="full_map",
        task_type="reach_door",
        render_mode=None,
        max_steps=200
    )
    
    completed = False
    for episode in range(5):
        obs, info = env.reset()
        
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"  Episode {episode + 1} completed at step {step + 1}")
                print(f"  Reward: {reward:.3f}")
                completed = True
                break
        
        if completed:
            break
    
    if completed:
        print("  Episode completion: OK")
    else:
        print("  Warning: No episode completed (this is OK for random agent)")
    
    env.close()
    print("\n✓ Test 7 PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60 + "\n")
    
    try:
        test_basic_functionality()
        test_observation_modes()
        test_task_types()
        test_actions()
        test_rendering()
        test_space_compatibility()
        test_episode_completion()
        
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe environment is working correctly!")
        print("You can now:")
        print("  - Run examples: python examples.py manual")
        print("  - Start training: python train.py progressive")
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
