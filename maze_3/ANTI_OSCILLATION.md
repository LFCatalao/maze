# Anti-Oscillation Reward System

## Problem

The agent was exploiting the distance-based reward system by oscillating back and forth:
- Move toward objective → get +2 reward for getting closer
- Move away from objective → move back toward objective → get +2 reward again
- Repeat infinitely to farm unlimited rewards without making real progress

This is a common issue in RL where distance-based rewards can be gamed.

## Solution Overview

The new reward system implements **four complementary anti-exploitation mechanisms**:

### 1. Best Distance Tracking ⭐ (Primary Solution)

**Problem it solves**: Agent farming rewards by repeatedly getting closer to the same distance.

**How it works**:
- Track the best (minimum) distance ever achieved to the current objective
- Only give distance reward when agent beats this record
- Reset best distance when entering a new phase (picking up key, opening door)

**Example**:
```
Distance 10 → 9: +1.0 reward (new best!)
Distance 9 → 10: no reward (worse than best)
Distance 10 → 9: no reward (not better than best of 9)
Distance 9 → 8: +1.0 reward (new best!)
```

**Code location**: `self.best_distance` tracking in `SimpleRewardWrapper`

---

### 2. Revisit Penalties

**Problem it solves**: Agent moving back and forth between the same positions.

**How it works**:
- Track last 10 positions in `position_history`
- When agent revisits a position, apply exponential penalty
- First revisit: -0.5, second revisit: -1.0, third: -1.5, etc.

**Example**:
```
Visit position A: normal rewards
Visit position B: normal rewards  
Visit position A again: -0.5 penalty (first revisit)
Visit position A again: -1.0 penalty (second revisit)
```

**Hyperparameter**: `PENALTY_REVISIT = -0.5`

---

### 3. Direction Change Penalties

**Problem it solves**: Agent rapidly changing directions (indicates confusion/oscillation).

**How it works**:
- Track the last movement direction
- Penalize when direction changes
- Stronger penalty for direct reversals (180° turns)

**Penalties**:
- Direction change (turn): `-0.1`
- Direct reversal (go back): `-0.3`

**Example**:
```
Move LEFT → Move LEFT: no penalty (momentum)
Move LEFT → Move UP: -0.1 penalty (90° turn)
Move LEFT → Move RIGHT: -0.3 penalty (180° reversal)
```

**Hyperparameters**: 
- `PENALTY_DIRECTION_CHANGE = -0.1`
- `PENALTY_REVERSAL = -0.3`

---

### 4. Position History Reset on Phase Change

**Problem it solves**: Prevents false penalties when agent legitimately needs to revisit areas.

**How it works**:
- When agent picks up a key or opens a door, clear position history
- This allows agent to freely navigate in the new phase without penalties
- Recognizes that the task structure has fundamentally changed

---

## Hyperparameters

All reward values are configurable at the top of `reward_wrappers.py`:

```python
# Main rewards
REWARD_GOAL = 100.0              # Reaching the goal
REWARD_CLOSER = 1.0              # NEW best distance to objective
REWARD_KEY_PICKUP = 20.0         # Picking up a key
REWARD_DOOR_OPEN = 20.0          # Opening a locked door
REWARD_EXPLORATION = 0.5         # Discovering a new cell

# Penalties
PENALTY_STEP = -0.1              # Time penalty per step
PENALTY_REVISIT = -0.5           # Per revisit (exponential)
PENALTY_DIRECTION_CHANGE = -0.1  # Changing direction
PENALTY_REVERSAL = -0.3          # 180° reversal
```

## Testing

Run the anti-oscillation test:

```bash
cd maze_3
python test_anti_oscillation.py
```

This demonstrates:
1. **Oscillation pattern**: Shows heavy penalties for back-and-forth movement
2. **Progressive movement**: Shows rewards for consistent progress toward goal

## Expected Behavior

### ❌ Bad: Oscillating Between Two Positions
```
Step 1: Move LEFT   → Reward: -0.1 (just step penalty, no distance improvement)
Step 2: Move RIGHT  → Reward: -0.7 (step + reversal + no distance reward)
Step 3: Move LEFT   → Reward: -1.2 (step + reversal + revisit)
Step 4: Move RIGHT  → Reward: -1.2 (step + reversal + revisit)
Total: -3.2 (very negative!)
```

### ✅ Good: Progressive Movement Toward Goal
```
Step 1: Move LEFT   → Reward: +1.4 (step + exploration + distance)
Step 2: Move LEFT   → Reward: +1.4 (step + exploration + distance)
Step 3: Move LEFT   → Reward: +1.4 (step + exploration + distance)
Step 4: Reach GOAL  → Reward: +100.0
Total: +104.2 (very positive!)
```

## Tuning Recommendations

### If agent is too cautious (not moving):
- Reduce `PENALTY_STEP` (currently -0.1)
- Increase `REWARD_CLOSER` (currently 1.0)
- Increase `REWARD_EXPLORATION` (currently 0.5)

### If agent still oscillates:
- Increase `PENALTY_REVISIT` (currently -0.5)
- Increase `PENALTY_REVERSAL` (currently -0.3)
- Decrease position history length (`max_history_length`, currently 10)

### If agent gets stuck:
- Decrease directional penalties
- Increase exploration reward
- Consider if best_distance is being reset appropriately

## Implementation Details

### Distance Calculation

Uses BFS (Breadth-First Search) to calculate actual pathfinding distance, not just Euclidean:
- Accounts for walls and locked doors
- Returns `float('inf')` if no path exists
- Only rewards when path exists AND beats best distance

### Phase Management

The reward system recognizes three phases:
1. **Finding key**: Target = accessible key position
2. **Opening door**: Target = locked door matching carried key
3. **Reaching goal**: Target = goal position

Best distance resets when transitioning between phases.

### Observation Modes

Works with both observation wrappers:
- `SimpleObs`: Compact 34-feature vector
- `EnhancedObs`: Full spatial awareness with explored map

## Related Files

- `environments/reward_wrappers.py`: Main implementation
- `environments/base_env.py`: Base environment
- `test_anti_oscillation.py`: Demonstration script
- `REWARD_SYSTEM.md`: General reward system documentation

## Summary

The anti-oscillation system prevents reward exploitation through:
1. ⭐ **Best distance tracking** - only reward actual progress
2. 🔄 **Revisit penalties** - discourage going to same positions
3. 🎯 **Direction penalties** - discourage rapid direction changes
4. 🔄 **Smart resets** - clear history when task changes

This creates a reward structure that genuinely rewards efficient goal-reaching behavior while heavily penalizing wasteful oscillation.
