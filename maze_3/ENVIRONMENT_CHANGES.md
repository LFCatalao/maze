# Environment Changes - Omnidirectional Movement with Fog of War

## Summary of Changes

The environment in `maze_3` has been updated with the following major changes:

### 1. **Omnidirectional Movement (4 Actions)**
   - **Old**: 5 actions (turn left, turn right, move forward, pickup, toggle)
   - **New**: 4 actions (up, down, left, right)
   - The agent can now move in any direction without needing to turn
   - Actions are defined as:
     - `Actions.UP = 0`
     - `Actions.DOWN = 1`
     - `Actions.LEFT = 2`
     - `Actions.RIGHT = 3`

### 2. **Auto-Pickup and Auto-Toggle**
   - **Keys**: Automatically picked up when the agent moves onto them
   - **Doors**: 
     - Automatically opened if closed (unlocked)
     - Automatically unlocked and opened if locked AND agent has the matching key
     - Agent cannot pass through locked doors without the correct key
   - No separate pickup or toggle actions needed

### 3. **7x7 Partial View Only**
   - The agent now **always** has a 7x7 partial view centered on its position
   - Previous observation modes (full_map, agent_view) are no longer used
   - The partial view shows what the agent can currently see

### 4. **Fog of War / Memory**
   - Added `explored_map` to the observation space
   - Tracks everything the agent has seen throughout the episode
   - Initially dark/unexplored, reveals as agent moves around
   - Shape: (19, 19, 3) - same as full grid
   - Allows agents to build a mental map of the environment

### 5. **Removed Direction Tracking**
   - Agent no longer has a "facing direction" since movement is omnidirectional
   - Removed `agent_dir` from environment state
   - Removed `direction` from observation space

## New Observation Space

```python
observation_space = spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8),
    "carrying": spaces.Discrete(7),  # 0-5 for key colors, 6 for nothing
    "explored_map": spaces.Box(low=0, high=255, shape=(19, 19, 3), dtype=np.uint8),
})
```

## Files Modified

1. **`maze_3/environments/base_env.py`**
   - Main environment file with all the changes above
   - Modified action handling, observation space, and rendering

## New Files Created

1. **`maze_3/manual_control.py`**
   - Interactive script for manual testing with keyboard controls
   - Shows dual view: explored map (with fog of war) + 7x7 partial view
   - Controls:
     - Arrow keys: Move up/down/left/right
     - R: Reset environment
     - ESC: Quit

2. **`maze_3/test_new_env.py`**
   - Automated test suite verifying all new mechanics
   - Tests:
     - Omnidirectional movement
     - Auto-pickup functionality
     - Auto door opening/unlocking
     - Partial view and fog of war
     - Environment 6 configuration

## Usage Examples

### Manual Testing
```bash
cd maze_3
python manual_control.py
```

This will launch Environment 6 with:
- Agent starting in bottom-left room
- Goal in top-right room
- 3 locked doors with keys placed strategically
- Visual display of both explored map and partial view

### Automated Testing
```bash
cd maze_3
python test_new_env.py
```

Runs all automated tests to verify functionality.

### Using in Training Code

```python
from environments.base_env import LockedRoomEnv, Actions

# Create environment with new mechanics
env = LockedRoomEnv(
    size=19,
    fixed_agent_pos=(16, 2),
    fixed_goal_pos=(2, 15),
    defined_doors=[
        {'door_idx': 2, 'key_pos': (16, 3)},
        {'door_idx': 4, 'key_pos': (2, 2)},
        {'door_idx': 3, 'key_pos': (10, 3)},
    ],
    verbose=True
)

obs, info = env.reset()

# obs now contains:
# - obs['image']: 7x7x3 partial view
# - obs['carrying']: what key agent is holding (0-5 or 6 for nothing)
# - obs['explored_map']: 19x19x3 map of what has been seen

# Take actions
obs, reward, terminated, truncated, info = env.step(Actions.UP)
obs, reward, terminated, truncated, info = env.step(Actions.RIGHT)
# etc.
```

## Environment 6 Configuration

The test configuration includes:
- **Agent Start**: (16, 2) - Bottom-left room
- **Goal**: (2, 15) - Top-right room
- **Doors** (3 locked doors):
  - Door 2 (Blue) at (15, 7) - Bottom-left door, key at (16, 3)
  - Door 4 (Grey) at (9, 11) - Middle-right door, key at (2, 2)
  - Door 3 (Purple) at (3, 11) - Top-right door, key at (10, 3)

## Training Implications

These changes will affect how agents need to learn:

1. **Simpler action space** (4 vs 5 actions) - easier to learn
2. **No direction management** - reduces state complexity
3. **Automatic key/door interaction** - agent doesn't need to learn pickup/toggle timing
4. **Partial observability** - agent must learn to navigate with limited vision
5. **Memory required** - agent benefits from remembering explored areas

The `explored_map` in observations allows agents to potentially use the full history of what they've seen, which could be very beneficial for navigation and planning.

## Backwards Compatibility

⚠️ **Warning**: These changes are **NOT backwards compatible** with previous training code or saved models that used:
- 5 actions
- Direction-based movement
- Full map observations
- Manual pickup/toggle actions

You will need to retrain agents from scratch with the new environment.

## Testing Results

All automated tests pass successfully:
- ✓ Omnidirectional movement works
- ✓ Auto-pickup works
- ✓ Auto door unlock/open works
- ✓ Partial view is 7x7
- ✓ Fog of war initialized and updated
- ✓ Exploration updates with movement
- ✓ Environment 6 configuration works
