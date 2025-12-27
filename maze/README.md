# Locked Room RL Environment

A flexible reinforcement learning environment based on MiniGrid's LockedRoom, with customizable observation spaces, progressive task learning, and pygame visualization.

## Features

- **Multiple Observation Modes**: Full map, partial view, or first-person agent view
- **Progressive Task Learning**: Train on simpler subtasks before the full task
- **Pygame Visualization**: See what's happening in real-time
- **Console Rendering**: Alternative ASCII visualization
- **Gymnasium Compatible**: Works with standard RL libraries
- **Flexible Task Definition**: Easily modify and create new tasks

## Installation

```bash
pip install gymnasium pygame numpy stable-baselines3
```

## Quick Start

### 1. Random Agent Demo

```bash
python examples.py random
```

### 2. Manual Control

```bash
python examples.py manual
```

**Controls:**
- Arrow Keys: Turn left/right, move forward
- SPACE: Pick up object
- E: Toggle/activate (open door)
- R: Reset environment
- ESC: Quit

### 3. Progressive Training Demo

```bash
python examples.py progress
```

### 4. Compare Observation Modes

```bash
python examples.py compare
```

## Environment Details

### Observation Modes

1. **Full Map** (`observation_mode="full_map"`)
   - Agent sees the entire grid (omniscient view)
   - Best for initial learning
   - Shape: (19, 19, 3)

2. **Partial View** (`observation_mode="partial_view"`)
   - Agent sees a local window around itself
   - More realistic than full map
   - Shape: (2*view_size+1, 2*view_size+1, 3)

3. **Agent View** (`observation_mode="agent_view"`)
   - First-person view (what's directly in front)
   - Most challenging
   - Shape: (7, 7, 3)

### Task Types

1. **Reach Door** (`task_type="reach_door"`)
   - Simplest task: just reach any door
   - Good for learning basic movement

2. **Get Key** (`task_type="get_key"`)
   - Pick up the key for the locked room
   - Learns pickup action

3. **Unlock Door** (`task_type="unlock_door"`)
   - Get the key and unlock the specific door
   - Combines navigation and interaction

4. **Full Task** (`task_type="full"`)
   - Complete mission: get key, unlock door, reach goal
   - Full complexity

### Action Space

| Action | Name | Description |
|--------|------|-------------|
| 0 | LEFT | Turn left |
| 1 | RIGHT | Turn right |
| 2 | FORWARD | Move forward |
| 3 | PICKUP | Pick up object |
| 4 | DROP | Drop object (unused) |
| 5 | TOGGLE | Toggle/activate object (open door) |


### Observation Space

The observation is a dictionary containing:
- **image**: Grid representation (object, color, state per cell)
- **direction**: Agent's facing direction (0-3)
- **carrying**: What the agent is carrying (color index or null)
- **mission**: Text description of the task (available but not in observation_space for compatibility)

### Rewards

- Success: `1.0 - 0.9 * (step_count / max_steps)`
- Failure: `0`

The reward encourages faster completion while still providing positive reinforcement.

## Usage Examples

### Basic Environment Creation

```python
from locked_room_env import LockedRoomEnv

# Create environment with full map view
env = LockedRoomEnv(
    observation_mode="full_map",
    task_type="full",
    render_mode="human",
    max_steps=500
)

# Reset and run
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Progressive Training

```python
# Stage 1: Learn to reach doors
env1 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="reach_door",
    max_steps=100
)

# Train your agent on env1...

# Stage 2: Learn to get keys
env2 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="get_key",
    max_steps=200
)

# Train your agent on env2...

# Stage 3: Full task
env3 = LockedRoomEnv(
    observation_mode="full_map",
    task_type="full",
    max_steps=500
)

# Train your agent on env3...
```

### Changing Observation During Training

```python
# Start with full map
env = LockedRoomEnv(observation_mode="full_map", task_type="full")

# Train for a while...

# Switch to partial view
env.observation_mode = "partial_view"
env._setup_observation_space()

# Continue training...

# Switch to agent view
env.observation_mode = "agent_view"
env._setup_observation_space()

# Final training...
```

## Training with Stable-Baselines3

### Single Task Training

```bash
python train.py single full
```

### Progressive Training

```bash
python train.py progressive
```

This will train three models:
1. `stage1_reach_door.zip` - Learns to reach doors
2. `stage2_get_key.zip` - Learns to pick up keys
3. `stage3_full_task.zip` - Learns the complete task

### Evaluate Trained Model

```bash
python train.py evaluate stage3_full_task
```

### Custom Training Script

```python
from locked_room_env import LockedRoomEnv
from stable_baselines3 import PPO

# Create environment
env = LockedRoomEnv(
    observation_mode="full_map",
    task_type="full",
    render_mode=None
)

# Flatten observation for standard RL algorithms
from train import FlattenObservation
env = FlattenObservation(env)

# Create and train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Save model
model.save("my_model")

# Evaluate
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Environment Customization

### Modify Grid Size

```python
env = LockedRoomEnv(size=25)  # Larger environment
```

### Change Maximum Steps

```python
env = LockedRoomEnv(max_steps=1000)  # More time to complete task
```

### Adjust Agent View Size

```python
env = LockedRoomEnv(
    observation_mode="agent_view",
    agent_view_size=5  # 5x5 view instead of 7x7
)
```

### Create Custom Tasks

You can extend the environment to create new tasks:

```python
class CustomLockedRoomEnv(LockedRoomEnv):
    def _check_task_completion(self):
        # Define your own success criteria
        if self.task_type == "custom_task":
            # Your logic here
            if some_condition:
                reward = 1.0
                terminated = True
                return terminated, reward
        
        # Fall back to parent implementation
        return super()._check_task_completion()
```

## File Structure

```
locked_room_env.py  - Main environment implementation
examples.py         - Example usage and demos
train.py           - Training scripts with stable-baselines3
README.md          - This file
```

## Visualization

### Pygame Rendering

- **Grey squares**: Empty space
- **Dark grey squares**: Walls
- **Colored squares**: Doors (darker border = locked)
- **Colored circles**: Keys
- **Green square**: Goal
- **Red circle**: Agent
- **Yellow line**: Agent's direction
- **Small colored dot above agent**: Carried object

### Console Rendering

```
==================================================
Step: 42/500
Mission: get the red key, unlock the red door and go to the goal
Carrying: red key
Position: [15, 8], Direction: 2
==================================================
####################
#........#........#
#........D........#
#........#........#
#.....G..#........#
#........#........#
####################
#........#........#
#........#....<...#
#........D........#
####################
==================================================
```

## Tips for Training

1. **Start Simple**: Begin with `task_type="reach_door"` and `observation_mode="full_map"`

2. **Progressive Difficulty**: Gradually increase task complexity:
   - reach_door → get_key → unlock_door → full

3. **Reduce Observation**: Once the agent masters with full map, reduce observation:
   - full_map → partial_view → agent_view

4. **Curriculum Learning**: Save models at each stage and use them as initialization for the next stage

5. **Hyperparameter Tuning**: Adjust learning rate, batch size, and network architecture based on observation complexity

6. **Reward Shaping**: Consider adding intermediate rewards for subtasks if the full task is too sparse

## Common Issues

### Pygame Window Not Responding
Make sure to call `env.render()` inside your training loop or use `render_mode=None` for headless training.

### Training Not Converging
- Try simpler tasks first (progressive training)
- Increase training timesteps
- Try different observation modes
- Adjust reward structure

### Memory Issues
Use `render_mode=None` during training to avoid creating pygame windows for each environment instance.

## License

This environment is created for educational and research purposes.

## Acknowledgments

Based on the MiniGrid LockedRoom environment from the MiniGrid library.
