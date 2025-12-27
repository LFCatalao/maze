# LOCKED ROOM ENVIRONMENT - ARCHITECTURE & DESIGN

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LockedRoomEnv                                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Configuration                          │  │
│  │  • observation_mode: full_map / partial_view / agent_view│  │
│  │  • task_type: reach_door / get_key / unlock_door / full  │  │
│  │  • render_mode: human / console / rgb_array / None       │  │
│  │  • max_steps: Maximum episode length                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    State Space                            │  │
│  │  • grid: (19, 19, 3) numpy array                         │  │
│  │    - [:,:,0] = Object type (wall, door, key, goal, etc.) │  │
│  │    - [:,:,1] = Color index (red, green, blue, etc.)      │  │
│  │    - [:,:,2] = State (open/closed/locked for doors)      │  │
│  │  • agent_pos: [y, x] position in grid                    │  │
│  │  • agent_dir: 0=right, 1=down, 2=left, 3=up             │  │
│  │  • carrying: Color index of carried object or None       │  │
│  │  • step_count: Current step number                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Action Space                            │  │
│  │  Discrete(7):                                             │  │
│  │    0 = LEFT      (turn left)                             │  │
│  │    1 = RIGHT     (turn right)                            │  │
│  │    2 = FORWARD   (move forward)                          │  │
│  │    3 = PICKUP    (pick up object)                        │  │
│  │    4 = DROP      (unused)                                │  │
│  │    5 = TOGGLE    (open/close door)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Observation Space                         │  │
│  │  Dict (validated fields):                                 │  │
│  │    • image: Grid view (shape depends on mode)            │  │
│  │    • direction: Discrete(4) - agent facing direction     │  │
│  │    • carrying: Discrete(7) - what agent holds            │  │
│  │  Additional info (returned but not validated):           │  │
│  │    • mission: Text - task description string             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Rendering System                         │  │
│  │  • Pygame: Visual 2D grid with colors                    │  │
│  │  • Console: ASCII text representation                    │  │
│  │  • RGB Array: Numpy array for recording                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Grid Layout

```
The environment consists of 6 rooms in a 2x3 layout:

┌─────────────────┬─────────────────┬─────────────────┐
│                 │                 │                 │
│   Room 1        D1    Room 2      D2    Room 3     │
│   (LOCKED)      │                 │                 │
│   [Goal]        │                 │                 │
│                 │                 │                 │
├─────────────────┼─────────────────┼─────────────────┤
│                 │                 │                 │
│   Room 4        D3    Room 5      D4    Room 6     │
│   [Key]         │                 │      [Agent]   │
│                 │                 │                 │
│                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘

Legend:
  D1, D2 = Doors (various states)
  D1 = LOCKED door (requires key to open)
  [Goal] = Target location
  [Key] = Key object (matches locked door color)
  [Agent] = Starting position
```

## Observation Modes

### 1. Full Map (observation_mode="full_map")
```
Agent sees the entire 19x19 grid (omniscient view)

Observation Shape: (19, 19, 3)

Use Case:
✓ Initial training
✓ Debugging
✓ Understanding environment structure
✓ Teaching basic navigation

Advantage: Complete information
Disadvantage: Not realistic
```

### 2. Partial View (observation_mode="partial_view")
```
Agent sees a local window around itself

         +-------+
         | | | | |
         +-------+
         | | A | |    A = Agent
         +-------+    View size: configurable
         | | | | |
         +-------+

Observation Shape: (2*view_size+1, 2*view_size+1, 3)
Default: (15, 15, 3) for view_size=7

Use Case:
✓ Intermediate difficulty
✓ More realistic than full map
✓ Learning to explore

Advantage: Partial information, still manageable
Disadvantage: May need memory to track visited areas
```

### 3. Agent View (observation_mode="agent_view")
```
Agent sees only what's in front (first-person view)

       Agent facing right →
       
       +-------+
       |   |   |
       +---A---+    Only sees the 7x7 grid ahead
       |   ↓   |
       +-------+

Observation Shape: (7, 7, 3)

Use Case:
✓ Most challenging setting
✓ Realistic robotics scenario
✓ Final evaluation

Advantage: Most realistic
Disadvantage: Requires strong spatial reasoning
```

## Task Types (Progressive Difficulty)

### Level 1: Reach Door (task_type="reach_door")
```
Task: Navigate to any door
Completion: Agent faces a door
Difficulty: ★☆☆☆☆

Skills Learned:
• Basic movement (forward, left, right)
• Spatial navigation
• Understanding grid structure

Success Criteria:
  Agent's forward cell contains a door
```

### Level 2: Get Key (task_type="get_key")
```
Task: Find and pick up the key
Completion: Agent holds the correct key
Difficulty: ★★☆☆☆

Skills Learned:
• Object detection
• PICKUP action
• Multi-step planning

Success Criteria:
  agent.carrying == locked_door_color
```

### Level 3: Unlock Door (task_type="unlock_door")
```
Task: Get key and unlock the locked door
Completion: Locked door is opened
Difficulty: ★★★☆☆

Skills Learned:
• Sequential actions
• Key-door matching
• TOGGLE action

Success Criteria:
  locked_door.state == OPEN (0)
```

### Level 4: Full Task (task_type="full")
```
Task: Get key, unlock door, reach goal
Completion: Agent reaches goal position
Difficulty: ★★★★★

Skills Learned:
• Complete task execution
• Long-term planning
• All previous skills combined

Success Criteria:
  agent.position == goal.position
```

## Reward Structure

```
Reward Function:
  reward = 1.0 - 0.9 * (step_count / max_steps)

Examples:
  Complete in 10 steps:   reward ≈ 0.98
  Complete in 50 steps:   reward ≈ 0.91
  Complete in 100 steps:  reward ≈ 0.82
  Complete in 250 steps:  reward ≈ 0.55
  Complete in 500 steps:  reward ≈ 0.10
  Fail to complete:       reward = 0.0

Benefits:
  ✓ Encourages fast completion
  ✓ Still rewards any success
  ✓ Dense feedback (time penalty)
  ✓ Easy to understand
```

## Training Strategy: Progressive Curriculum

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

Stage 1: REACH DOOR
├─ Observation: full_map
├─ Task: reach_door
├─ Max Steps: 100
├─ Training: 50k steps
└─ Success Criteria: >80% success rate
          ↓
Stage 2: GET KEY
├─ Observation: full_map
├─ Task: get_key
├─ Max Steps: 200
├─ Training: 100k steps
├─ Load: Stage 1 model (optional)
└─ Success Criteria: >70% success rate
          ↓
Stage 3: UNLOCK DOOR
├─ Observation: partial_view
├─ Task: unlock_door
├─ Max Steps: 300
├─ Training: 150k steps
├─ Load: Stage 2 model (optional)
└─ Success Criteria: >60% success rate
          ↓
Stage 4: FULL TASK
├─ Observation: partial_view or agent_view
├─ Task: full
├─ Max Steps: 500
├─ Training: 200k+ steps
├─ Load: Stage 3 model (optional)
└─ Success Criteria: >50% success rate
          ↓
Final Model Ready!
```

## File Structure & Dependencies

```
locked_room_env/
│
├── locked_room_env.py      Main environment implementation
│   ├── LockedRoomEnv       Core environment class
│   ├── Objects             Enum for object types
│   ├── Colors              Enum for colors
│   └── Actions             Enum for actions
│
├── examples.py             Demo and manual control
│   ├── random_agent_demo() Random agent visualization
│   ├── manual_control_demo() Keyboard control
│   ├── progressive_training_demo() Show progression
│   └── compare_observation_modes() Compare modes
│
├── train.py               Training utilities
│   ├── FlattenObservation  Wrapper for standard RL
│   ├── train_progressive() Progressive curriculum
│   ├── train_single_task() Single task training
│   ├── evaluate_model()    Model evaluation
│   └── custom_training_loop() Custom RL implementation
│
├── test_env.py            Test suite
│   ├── test_basic_functionality()
│   ├── test_observation_modes()
│   ├── test_task_types()
│   ├── test_actions()
│   └── test_rendering()
│
├── QUICK_START.py         Quick start examples
├── README.md              Full documentation
└── requirements.txt       Dependencies
```

## Integration with RL Libraries

### Stable-Baselines3
```python
from stable_baselines3 import PPO, A2C, DQN
from train import FlattenObservation

env = LockedRoomEnv(...)
env = FlattenObservation(env)  # Flatten dict observation

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100_000)
```

### Custom Algorithms
```python
env = LockedRoomEnv(...)

# Access raw observations
obs, info = env.reset()
# obs is a dict with 'image', 'direction', 'carrying', 'mission'

# You can process this however your algorithm needs
image_flat = obs['image'].flatten()
direction_onehot = one_hot(obs['direction'], 4)
carrying_onehot = one_hot(obs['carrying'], 7)
```

## Key Implementation Details

### Grid Encoding
```
Each cell: [object_type, color_index, state]

Object Types:
  0 = EMPTY
  1 = WALL
  2 = DOOR
  3 = KEY
  4 = GOAL
  5 = AGENT (not stored in grid)

Colors:
  0 = RED, 1 = GREEN, 2 = BLUE
  3 = PURPLE, 4 = YELLOW, 5 = GREY

States (for doors):
  0 = OPEN
  1 = CLOSED
  2 = LOCKED
```

### Action Execution
```
LEFT/RIGHT:   Change agent_dir (no movement)
FORWARD:      Move in agent_dir if cell is free
PICKUP:       Pick up key in front cell
TOGGLE:       Open/unlock door in front cell
              (needs matching key for locked doors)
```

### Rendering
```
Pygame:
  - Cells are 32x32 pixels
  - Walls: Dark grey rectangles
  - Doors: Colored rectangles (black lock if locked)
  - Keys: Colored circles
  - Goal: Green rectangle
  - Agent: Red circle with direction indicator
  
Console:
  - ASCII grid representation
  - #=wall, D=door, K=key, G=goal, .=empty
  - Agent shown as arrow: > v < ^
```

## Performance Considerations

### Training Speed
- `render_mode=None` for fastest training
- Use vectorized environments for parallel training
- Smaller observation modes train faster

### Memory
- Full map: ~1KB per observation
- Partial view: ~600 bytes per observation
- Agent view: ~150 bytes per observation

### Typical Training Times (on CPU)
```
Task          | Timesteps | Time (approx)
--------------|-----------|---------------
reach_door    | 50k       | 5-10 minutes
get_key       | 100k      | 10-20 minutes
unlock_door   | 150k      | 15-30 minutes
full          | 200k+     | 30-60 minutes
```

## Customization Examples

### Change Grid Size
```python
env = LockedRoomEnv(size=25)  # Larger environment
```

### Custom Reward
```python
class CustomRewardEnv(LockedRoomEnv):
    def _check_task_completion(self):
        terminated, reward = super()._check_task_completion()
        # Modify reward
        reward *= 2.0  # Double reward
        return terminated, reward
```

### Add New Objects
```python
class ExtendedEnv(LockedRoomEnv):
    def _generate_locked_room(self):
        super()._generate_locked_room()
        # Add custom objects to self.grid
```

## Troubleshooting

### Agent not learning
- Start with easier task (reach_door)
- Use full_map observation
- Increase training timesteps
- Check reward signal is working

### Training too slow
- Use render_mode=None
- Reduce observation size
- Use GPU if available
- Parallelize with vectorized envs

### Pygame window issues
- Set render_mode=None for training
- Only use "human" mode for evaluation
- Use "console" mode for debugging

## Best Practices

1. **Start Simple**: Begin with full_map + reach_door
2. **Progressive Training**: Gradually increase difficulty
3. **Monitor Metrics**: Track success rate and average reward
4. **Save Checkpoints**: Save models at each stage
5. **Evaluate Regularly**: Test on held-out episodes
6. **Visualize**: Use pygame to understand agent behavior
7. **Debug**: Use console mode for quick checks
