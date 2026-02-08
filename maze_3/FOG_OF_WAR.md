# Fog of War Visualization

The evaluation and manual control scripts now support **Fog of War** mode, which shows only what the agent has explored/seen, just like the agent's memory.

## What is Fog of War?

**Fog of War** is a visualization mode where:
- **Black areas** = Unexplored (agent hasn't seen them yet)
- **White/Colored areas** = Explored (agent has seen them)

This matches exactly what the agent knows through its `explored_map` observation, giving you a true view of the agent's knowledge state.

## Usage

### Evaluate Trained Models

```bash
# Default: Fog of war ON (shows only explored areas)
python evaluate.py path/to/model

# Show full map (fog of war OFF)
python evaluate.py path/to/model --full-map
```

### Manual Control

```bash
# Default: Fog of war ON
python evaluate.py --manual

# Full map mode
python evaluate.py --manual --full-map

# Toggle fog of war during play: Press 'F' key
```

### Random Agent

```bash
# Default: Fog of war ON
python evaluate.py --random

# Full map mode
python evaluate.py --random --full-map
```

## Keyboard Controls (Manual Mode)

- **Arrow Keys**: Move Up/Down/Left/Right
- **R**: Reset environment
- **F**: Toggle fog of war ON/OFF
- **ESC**: Quit

## Implementation Details

### Base Environment (`base_env.py`)

Added `fog_of_war` parameter to rendering:

```python
def render(self, fog_of_war=False):
    """Render the environment
    
    Args:
        fog_of_war: If True, only show what agent has explored
    """
```

The renderer uses `self.explored_map` to determine what the agent has seen. Unexplored cells are rendered as black (30, 30, 30), while explored cells show their actual content.

### Evaluation Script (`evaluate.py`)

- Added `--full-map` flag to disable fog of war (default is ON)
- Manual control now supports 'F' key to toggle fog of war live
- All modes (evaluate, random, manual) support fog of war

### How Exploration Works

The environment tracks explored cells in `explored_map`:
- Updated every step based on the agent's 7x7 view
- Empty explored cells have `state=1` to distinguish from unexplored
- Picked-up keys and opened doors are remembered in the explored map

## Benefits

1. **Training Insight**: See exactly what the agent knows at each moment
2. **Debugging**: Identify if agent is missing important areas
3. **Realistic View**: Matches the partial observability the agent experiences
4. **Educational**: Understand how agents learn to explore and remember

## Example Output

```
Configuration:
  Environment: E5_FIXED_simple
  Reward: simple
  Observation: full_map
  Episodes: 10
  Render: True
  Fog of War: True (showing only explored areas)
```

As the agent moves, you'll see the fog gradually clear, revealing the map as the agent discovers it!
