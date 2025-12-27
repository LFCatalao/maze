"""
QUICK START GUIDE - Locked Room RL Environment
================================================

This guide shows you how to get started quickly with the environment.
"""

# =============================================================================
# INSTALLATION
# =============================================================================

"""
1. Install dependencies:
   pip install gymnasium pygame numpy stable-baselines3

2. All files are in the same directory:
   - locked_room_env.py  (main environment)
   - examples.py         (demo scripts)
   - train.py           (training scripts)
   - test_env.py        (tests)
"""

# =============================================================================
# EXAMPLE 1: Basic Usage
# =============================================================================

from locked_room_env import LockedRoomEnv

# Create environment
env = LockedRoomEnv(
    observation_mode="full_map",    # Agent sees entire grid
    task_type="full",                # Complete task
    render_mode="human",             # Pygame visualization
    max_steps=500                    # Maximum steps per episode
)

# Reset and run
obs, info = env.reset()
print(f"Mission: {obs['mission']}")

for step in range(100):
    # Take random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render
    env.render()
    
    # Check if done
    if terminated or truncated:
        print(f"Episode finished! Reward: {reward:.3f}")
        break

env.close()

# =============================================================================
# EXAMPLE 2: Different Observation Modes
# =============================================================================

# Full map - Agent sees everything
env_full = LockedRoomEnv(observation_mode="full_map", task_type="full")

# Partial view - Agent sees local area
env_partial = LockedRoomEnv(observation_mode="partial_view", task_type="full")

# Agent view - First-person perspective
env_agent = LockedRoomEnv(observation_mode="agent_view", task_type="full")

# =============================================================================
# EXAMPLE 3: Progressive Task Learning
# =============================================================================

# Stage 1: Learn basic movement (reach any door)
env_stage1 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    max_steps=100
)

# Train agent on stage 1...
# Once it succeeds consistently, move to stage 2

# Stage 2: Learn to pick up objects (get the key)
env_stage2 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="get_key",
    max_steps=200
)

# Train agent on stage 2...
# Once it succeeds consistently, move to stage 3

# Stage 3: Learn to use keys (unlock door)
env_stage3 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="unlock_door",
    max_steps=300
)

# Train agent on stage 3...
# Once it succeeds consistently, move to stage 4

# Stage 4: Complete full task
env_stage4 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="full",
    max_steps=500
)

# =============================================================================
# EXAMPLE 4: Training with Stable-Baselines3
# =============================================================================

from stable_baselines3 import PPO
from train import FlattenObservation

# Create and wrap environment
env = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    render_mode=None  # No rendering during training
)
env = FlattenObservation(env)  # Flatten observation for RL algorithm

# Create model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=50_000)

# Save
model.save("my_trained_agent")

# Evaluate
env_eval = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    render_mode="human"
)
env_eval = FlattenObservation(env_eval)

obs, info = env_eval.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_eval.step(action)
    env_eval.render()
    
    if terminated or truncated:
        print(f"Task completed! Reward: {reward:.3f}")
        break

env_eval.close()

# =============================================================================
# EXAMPLE 5: Manual Control (Interactive)
# =============================================================================

"""
Run the manual control demo:
    python examples.py manual

Controls:
    Arrow Keys: Turn left/right, move forward
    SPACE: Pick up object
    E: Toggle/activate (open door)
    R: Reset environment
    ESC: Quit
"""

# =============================================================================
# EXAMPLE 6: Custom Training Loop
# =============================================================================

env = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    render_mode=None
)

# Training loop
for episode in range(100):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(100):
        # Your policy here (random for demo)
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Here you would update your policy
        # Example: store transition (obs, action, reward, next_obs, done)
        # Then train your network on these transitions
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}")

env.close()

# =============================================================================
# EXAMPLE 7: Accessing Environment State
# =============================================================================

env = LockedRoomEnv(observation_mode="full_map", task_type="full")
obs, info = env.reset()

# Observation dictionary
print("Observation keys:", obs.keys())
print("Image shape:", obs['image'].shape)        # Grid representation
print("Direction:", obs['direction'])             # 0=right, 1=down, 2=left, 3=up
print("Carrying:", obs['carrying'])               # What agent holds
print("Mission:", obs['mission'])                 # Task description

# Info dictionary
print("Info keys:", info.keys())
print("Agent position:", info['agent_pos'])       # [y, x]
print("Step count:", info['step_count'])          # Current step

# Environment internals
print("Grid shape:", env.grid.shape)              # (19, 19, 3)
print("Door positions:", env.door_positions)      # Dict of door locations
print("Key positions:", env.key_positions)        # Dict of key locations
print("Goal position:", env.goal_pos)             # Goal location

env.close()

# =============================================================================
# EXAMPLE 8: Changing Observation Mode During Training
# =============================================================================

# Start with full map for easier learning
env = LockedRoomEnv(observation_mode="full_map", task_type="full")

# Train for 10,000 steps...

# Switch to partial view
env.observation_mode = "partial_view"
env._setup_observation_space()

# Continue training for 10,000 steps...

# Switch to agent view (hardest)
env.observation_mode = "agent_view"
env._setup_observation_space()

# Final training...

env.close()

# =============================================================================
# COMMON USAGE PATTERNS
# =============================================================================

# 1. Quick testing with console rendering
env = LockedRoomEnv(render_mode="console", max_steps=50)
obs, info = env.reset()
for _ in range(20):
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
env.close()

# 2. Training without visualization
env = LockedRoomEnv(render_mode=None)
# ... training code ...

# 3. Evaluation with visualization
env = LockedRoomEnv(render_mode="human")
# ... evaluation code ...

# 4. Custom grid size
env = LockedRoomEnv(size=25)  # Larger environment

# 5. Longer episodes
env = LockedRoomEnv(max_steps=1000)

# =============================================================================
# AVAILABLE ACTIONS
# =============================================================================

from locked_room_env import Actions

print("Available actions:")
print(f"  {Actions.LEFT} = Turn left")
print(f"  {Actions.RIGHT} = Turn right")
print(f"  {Actions.FORWARD} = Move forward")
print(f"  {Actions.PICKUP} = Pick up object")
print(f"  {Actions.DROP} = Drop object (unused)")
print(f"  {Actions.TOGGLE} = Toggle/activate (open door)")
print(f"  {Actions.DONE} = Done (unused)")

# =============================================================================
# TIPS FOR SUCCESS
# =============================================================================

"""
1. Start Simple:
   - Use observation_mode="full_map"
   - Use task_type="reach_door"
   - Train until agent succeeds consistently

2. Progress Gradually:
   - Increase task complexity: reach_door → get_key → unlock_door → full
   - Reduce observation: full_map → partial_view → agent_view

3. Monitor Training:
   - Print rewards and success rates
   - Use tensorboard for detailed logging
   - Save checkpoints frequently

4. Hyperparameters:
   - Start with default PPO settings
   - Increase batch_size for larger observations
   - Adjust learning_rate if not converging

5. Debugging:
   - Use render_mode="console" for quick checks
   - Print mission and observations
   - Verify agent can complete easier tasks first
"""
