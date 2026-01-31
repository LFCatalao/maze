"""
Simplified Reward Wrapper for RL Experiments

Single reward function with adjustable hyperparameters.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# =============================================================================
# HYPERPARAMETERS (adjust these for experiments)
# =============================================================================

REWARD_GOAL = 100.0
REWARD_CLOSER = 0.5
PENALTY_FURTHER = 0.0  # Removed - already captured by REWARD_CLOSER
PENALTY_STEP = -0.1
REWARD_TURN = 0.0
REWARD_KEY_PICKUP = 20.0
REWARD_DOOR_OPEN = 20.0

# New exploration rewards
REWARD_EXPLORATION = 0.3  # Bonus for seeing a new cell
PENALTY_EMPTY_ROOM = -0.2  # Penalty per step in fully explored empty room
REWARD_USEFUL_ROOM = 0.1  # Reward per step in room with objective


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_room(pos):
    """Determine which room the position is in."""
    y, x = pos

    if 7 <= x <= 11:
        return "corridor"

    if x < 7:
        if y < 6:
            return "left_top"
        elif y < 12:
            return "left_middle"
        else:
            return "left_bottom"
    else:
        if y < 6:
            return "right_top"
        elif y < 12:
            return "right_middle"
        else:
            return "right_bottom"


# =============================================================================
# OBSERVATION WRAPPER
# =============================================================================


class SimpleObs(gym.ObservationWrapper):
    """Simple observation with key support - updated for omnidirectional movement."""

    def __init__(self, env):
        super().__init__(env)
        # Reduced from 12 to 10 since we removed direction and direction-dependent fields
        self.observation_space = spaces.Box(0, 20, (10,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_y, agent_x = base_env.agent_pos
        goal_y, goal_x = base_env.goal_pos

        # Key info
        has_key = 1.0 if base_env.carrying is not None else 0.0

        # Key position
        key_y, key_x = -1.0, -1.0
        if hasattr(base_env, "key_positions") and base_env.key_positions:
            if base_env.carrying is None:
                key_pos = list(base_env.key_positions.keys())[0]
                key_y, key_x = float(key_pos[0]), float(key_pos[1])

        # Door position
        door_y, door_x = -1.0, -1.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_y, door_x = float(door_pos[0]), float(door_pos[1])

        return np.array(
            [
                agent_y,
                agent_x,
                goal_y,
                goal_x,
                has_key,
                key_y,
                key_x,
                door_y,
                door_x,
                0.0,  # Placeholder for compatibility
            ],
            dtype=np.float32,
        )


# =============================================================================
# REWARD WRAPPER
# =============================================================================


class SimpleRewardWrapper(gym.RewardWrapper):
    """
    Three-phase reward: key -> door -> goal
    All bonuses are one-time only to prevent exploitation.
    Includes exploration bonus and room utility penalties.
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        # One-time bonus flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        # Repeat action tracking
        self.last_action = None
        self.repeat_count = 0
        # Exploration tracking
        self.explored_cells = set()  # Track all cells agent has seen
        self.fully_explored_rooms = set()  # Track rooms that are fully explored
        self.room_contents = {}  # Cache what's in each room {room_idx: has_useful_item}

    def _get_base_env(self):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        return base_env

    def _get_current_target(self):
        """Return current target: accessible key -> door we can open -> goal."""
        base_env = self._get_base_env()

        # Phase 1: Not carrying a key -> find an ACCESSIBLE key (if any exist)
        if base_env.carrying is None:
            if hasattr(base_env, "key_positions") and base_env.key_positions:
                accessible_key = self._find_accessible_key(base_env)
                if accessible_key:
                    return accessible_key
            # No keys left -> target is goal
            return base_env.goal_pos

        # Phase 2: Carrying a key -> find the matching locked door
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos, (door_color, door_state) in base_env.door_positions.items():
                if door_state == 2 and door_color == base_env.carrying:
                    return door_pos

        # Phase 3: Have key but no matching locked door (shouldn't happen) -> goal
        return base_env.goal_pos

    def _find_accessible_key(self, base_env):
        """Find a key that is not behind a locked door."""
        # Build set of rooms blocked by locked doors
        blocked_rooms = set()

        # Door position to room index mapping
        door_to_room = {(3, 7): 0, (9, 7): 1, (15, 7): 2, (3, 11): 3, (9, 11): 4, (15, 11): 5}

        if hasattr(base_env, "door_positions"):
            for door_pos, (color, state) in base_env.door_positions.items():
                if state == 2:  # Locked
                    if door_pos in door_to_room:
                        blocked_rooms.add(door_to_room[door_pos])

        # Find a key not in a blocked room
        for key_pos, key_color in base_env.key_positions.items():
            key_room = self._get_room_index(key_pos)
            if key_room not in blocked_rooms:
                return key_pos

        # Fallback: return first key (shouldn't happen in well-designed envs)
        return list(base_env.key_positions.keys())[0]

    def _get_room_index(self, pos):
        """Get room index for a position (matches env logic)."""
        y, x = pos

        # Corridor
        if 7 <= x <= 11:
            return -1

        # Left side rooms
        if x < 7:
            if y <= 5:
                return 0
            if y <= 11:
                return 1
            return 2

        # Right side rooms
        if x > 11:
            if y <= 5:
                return 3
            if y <= 11:
                return 4
            return 5

        return -1

    def _is_room_fully_explored(self, room_idx):
        """Check if a room is fully explored by checking all its cells."""
        if room_idx == -1:  # Corridor
            return False

        # Define room bounds (y_start, y_end, x_start, x_end) inclusive
        bounds = {
            0: (1, 5, 1, 6),
            1: (7, 11, 1, 6),
            2: (13, 17, 1, 6),
            3: (1, 5, 12, 17),
            4: (7, 11, 12, 17),
            5: (13, 17, 12, 17)
        }

        if room_idx not in bounds:
            return False

        y1, y2, x1, x2 = bounds[room_idx]

        # Check if all cells in room are explored
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if (y, x) not in self.explored_cells:
                    return False

        return True

    def _room_has_useful_item(self, room_idx):
        """Check if room contains key, door, or goal."""
        if room_idx == -1:
            return False

        base_env = self._get_base_env()

        # Define room bounds
        bounds = {
            0: (1, 5, 1, 6),
            1: (7, 11, 1, 6),
            2: (13, 17, 1, 6),
            3: (1, 5, 12, 17),
            4: (7, 11, 12, 17),
            5: (13, 17, 12, 17)
        }

        if room_idx not in bounds:
            return False

        y1, y2, x1, x2 = bounds[room_idx]

        # Check for goal
        if base_env.goal_pos:
            gy, gx = base_env.goal_pos
            if y1 <= gy <= y2 and x1 <= gx <= x2:
                return True

        # Check for keys
        if hasattr(base_env, 'key_positions'):
            for (ky, kx), _ in base_env.key_positions.items():
                if y1 <= ky <= y2 and x1 <= kx <= x2:
                    return True

        # Check for doors (doors are at room entrances, we consider them part of the room)
        if hasattr(base_env, 'door_positions'):
            for (dy, dx), _ in base_env.door_positions.items():
                # Doors at boundaries - check if adjacent to this room
                if y1 <= dy <= y2 and x1 <= dx <= x2:
                    return True

        return False

    def _update_explored_cells(self, obs):
        """Update set of explored cells from observation."""
        if 'explored_map' not in obs:
            return 0

        explored_map = obs['explored_map']
        base_env = self._get_base_env()
        
        new_cells = 0
        for y in range(base_env.size):
            for x in range(base_env.size):
                # Cell is explored if it's not all zeros in explored_map
                if np.any(explored_map[y, x] > 0):
                    if (y, x) not in self.explored_cells:
                        self.explored_cells.add((y, x))
                        new_cells += 1

        return new_cells

    def _distance_to_target(self):
        """Distance to nearest adjacent cell of current target."""
        base_env = self._get_base_env()
        pos = base_env.agent_pos
        target = self._get_current_target()

        # For key and door, we need to be adjacent
        # For goal, we step on it directly
        if (
            base_env.carrying is None
            and hasattr(base_env, "key_positions")
            and base_env.key_positions
        ):
            # Key phase - distance to adjacent cell
            distances = [
                abs(pos[0] - (target[0] - 1)) + abs(pos[1] - target[1]),
                abs(pos[0] - (target[0] + 1)) + abs(pos[1] - target[1]),
                abs(pos[0] - target[0]) + abs(pos[1] - (target[1] - 1)),
                abs(pos[0] - target[0]) + abs(pos[1] - (target[1] + 1)),
            ]
            return min(distances)

        # Door phase - also need adjacent
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_state = base_env.grid[door_pos[0], door_pos[1], 2]
            if door_state == 2:  # Still locked
                distances = [
                    abs(pos[0] - (target[0] - 1)) + abs(pos[1] - target[1]),
                    abs(pos[0] - (target[0] + 1)) + abs(pos[1] - target[1]),
                    abs(pos[0] - target[0]) + abs(pos[1] - (target[1] - 1)),
                    abs(pos[0] - target[0]) + abs(pos[1] - (target[1] + 1)),
                ]
                return min(distances)

        # Goal phase - direct distance
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def _facing_target(self):
        """Check if moving toward current target (simplified for omnidirectional)."""
        # Since we have omnidirectional movement, this concept doesn't apply
        # We'll just return True to not break existing logic
        return True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._distance_to_target()
        # Reset all flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        self.last_action = None
        self.repeat_count = 0
        # Reset exploration tracking
        self.explored_cells = set()
        self.fully_explored_rooms = set()
        self.room_contents = {}
        # Initialize with current view
        self._update_explored_cells(obs)
        return obs, info

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)
        base_env = self._get_base_env()
        had_key_before = base_env.carrying is not None
        key_color_before = base_env.carrying

        # Track ALL door states before action
        door_states_before = {}
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos in base_env.door_positions:
                door_states_before[door_pos] = base_env.grid[door_pos[0], door_pos[1], 2]

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        moved = old_pos != new_pos
        has_key_now = base_env.carrying is not None

        # Goal reached
        if base_reward > 0:
            return obs, REWARD_GOAL, terminated, truncated, info

        reward = PENALTY_STEP

        # Exploration bonus - reward for seeing new cells
        new_cells_count = self._update_explored_cells(obs)
        if new_cells_count > 0:
            reward += REWARD_EXPLORATION * new_cells_count

        # Key pickup bonus
        if has_key_now and not had_key_before:
            reward += REWARD_KEY_PICKUP
            self.previous_distance = self._distance_to_target()
            return obs, reward, terminated, truncated, info

        # Door open bonus - check ALL doors
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos in base_env.door_positions:
                state_before = door_states_before.get(door_pos, 0)
                state_now = base_env.grid[door_pos[0], door_pos[1], 2]
                if state_before == 2 and state_now == 0:  # Was locked, now open
                    reward += REWARD_DOOR_OPEN
                    self.previous_distance = self._distance_to_target()
                    return obs, reward, terminated, truncated, info

        # Distance reward (only if moving closer/staying same distance)
        current_distance = self._distance_to_target()
        if current_distance < self.previous_distance:
            reward += REWARD_CLOSER
        # Removed penalty for getting further
        self.previous_distance = current_distance

        # Room utility reward/penalty
        current_room = self._get_room_index(new_pos)
        if current_room != -1:  # Not in corridor
            # Check if room is fully explored
            if self._is_room_fully_explored(current_room):
                if current_room not in self.fully_explored_rooms:
                    self.fully_explored_rooms.add(current_room)
                    # Cache whether room has useful items
                    self.room_contents[current_room] = self._room_has_useful_item(current_room)

                # Apply penalty/reward based on room contents
                if current_room in self.room_contents:
                    if self.room_contents[current_room]:
                        reward += REWARD_USEFUL_ROOM  # Reward for being in useful room
                    else:
                        reward += PENALTY_EMPTY_ROOM  # Penalty for wasting time in empty room

        # Penalty for repeating non-moving actions
        if action == self.last_action and not moved:
            self.repeat_count += 1
            if self.repeat_count > 3:
                reward -= 0.2
        else:
            self.repeat_count = 0
        self.last_action = action

        return obs, reward, terminated, truncated, info

    def reward(self, reward):
        return reward


# =============================================================================
# WRAPPER FACTORY
# =============================================================================

REWARD_WRAPPERS = {
    "simple": SimpleRewardWrapper,
}


def apply_reward_wrapper(env, reward_type="simple"):
    """Apply reward wrapper to environment."""
    if reward_type not in REWARD_WRAPPERS:
        raise ValueError(f"Unknown reward type: {reward_type}")

    return REWARD_WRAPPERS[reward_type](env)
