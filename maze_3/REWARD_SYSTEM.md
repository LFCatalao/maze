# Exploration-Based Reward System

## Overview
The reward system encourages efficient exploration and discourages wasteful behavior in empty rooms.

## Reward Components

### Base Rewards
- **REWARD_GOAL** = 100.0 - Reaching the goal
- **REWARD_KEY_PICKUP** = 20.0 - Picking up a key
- **REWARD_DOOR_OPEN** = 20.0 - Opening a locked door
- **PENALTY_STEP** = -0.1 - Cost of taking a step

### Distance-Based Rewards
- **REWARD_CLOSER** = 0.5 - Moving closer to current objective (key → door → goal)
- **PENALTY_FURTHER** = 0.0 (removed) - Getting further is already captured by not receiving REWARD_CLOSER

### Exploration Rewards (NEW)
- **REWARD_EXPLORATION** = 0.3 per new cell - Encourages curiosity and discovering unexplored areas
- **PENALTY_EMPTY_ROOM** = -0.2 per step - Discourages staying in fully explored rooms with no useful items
- **REWARD_USEFUL_ROOM** = 0.1 per step - Encourages being in rooms that contain keys, doors, or the goal

## Key Concepts

### Exploration Tracking
- The agent tracks every cell it has seen (via the 7x7 partial view)
- New cells grant an exploration bonus, encouraging systematic exploration
- This bonus is only awarded once per cell (no exploitation)

### Room Utility System
The agent learns to evaluate rooms based on their contents:

1. **Fully Explored Rooms**: Once all cells in a room are visible, the room is considered "fully explored"
2. **Useful Rooms**: Rooms containing:
   - Keys (that haven't been picked up)
   - Doors (locked or unlocked)
   - The goal position
3. **Empty Rooms**: Fully explored rooms with no useful items

### Reward Logic
1. Agent explores new areas → gets +0.3 per new cell
2. Agent enters a fully explored empty room → gets -0.2 per step (waste of time)
3. Agent stays in a room with objectives → gets +0.1 per step (good positioning)
4. Agent moves closer to objective → gets +0.5
5. Agent picks up key or opens door → gets +20.0
6. Agent reaches goal → gets +100.0

## Example Reward Sequence

```
Step 1-9:   +2.50  (Exploring new corridor: +0.3×9 cells - 0.1 step + other bonuses)
Step 10:    +22.00 (Key pickup: +20.0 + exploration)
Step 43-46: +0.40  (Moving closer to door: +0.5 - 0.1 step)
Step 87:    +2.50  (Exploring near door)
Step 88:    +22.00 (Door unlock: +20.0 + exploration)
Step 100:   +19.90 (Key pickup in previously explored area)
Step 174:   +1.50  (Final approach to goal with exploration)
```

## Benefits

1. **Encourages Systematic Exploration**: Agent is rewarded for discovering new areas
2. **Discourages Backtracking to Empty Rooms**: Penalty for revisiting fully explored empty spaces
3. **Prioritizes Objectives**: Strong rewards for keys/doors/goal keep agent focused
4. **No Negative Reinforcement for Exploration**: Removed penalty for moving away from objective
5. **Partial Observability Friendly**: Works well with 7x7 view since exploration bonus naturally guides discovery

## Training Tips

- The exploration bonus helps with sparse reward problems
- The empty room penalty prevents the agent from getting stuck in loops
- The useful room reward encourages the agent to stay near objectives
- Adjust hyperparameters based on maze complexity:
  - Larger mazes may need higher exploration rewards
  - More complex key chains may benefit from higher key/door rewards
