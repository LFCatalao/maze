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
PENALTY_FURTHER = -0.6
PENALTY_STEP = -0.1
REWARD_TURN = 0.0
REWARD_KEY_PICKUP = 20.0
REWARD_DOOR_OPEN = 20.0


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
    """Simple observation with key support."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 20, (12,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_y, agent_x = base_env.agent_pos
        goal_y, goal_x = base_env.goal_pos
        direction = base_env.agent_dir

        # Key info
        has_key = 1.0 if base_env.carrying is not None else 0.0

        # Key position
        key_y, key_x = -1.0, -1.0
        can_pickup = 0.0
        if hasattr(base_env, "key_positions") and base_env.key_positions:
            if base_env.carrying is None:
                key_pos = list(base_env.key_positions.keys())[0]
                key_y, key_x = float(key_pos[0]), float(key_pos[1])

                # Check if adjacent to key AND facing it
                # Direction: 0=right, 1=down, 2=left, 3=up
                if direction == 0 and agent_y == key_pos[0] and agent_x == key_pos[1] - 1:
                    can_pickup = 1.0  # Key is to the right, facing right
                elif direction == 1 and agent_y == key_pos[0] - 1 and agent_x == key_pos[1]:
                    can_pickup = 1.0  # Key is below, facing down
                elif direction == 2 and agent_y == key_pos[0] and agent_x == key_pos[1] + 1:
                    can_pickup = 1.0  # Key is to the left, facing left
                elif direction == 3 and agent_y == key_pos[0] + 1 and agent_x == key_pos[1]:
                    can_pickup = 1.0  # Key is above, facing up

        # Door position
        door_y, door_x = -1.0, -1.0
        can_open = 0.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_y, door_x = float(door_pos[0]), float(door_pos[1])

            # Check if adjacent to door AND facing it
            if direction == 0 and agent_y == door_pos[0] and agent_x == door_pos[1] - 1:
                can_open = 1.0
            elif direction == 1 and agent_y == door_pos[0] - 1 and agent_x == door_pos[1]:
                can_open = 1.0
            elif direction == 2 and agent_y == door_pos[0] and agent_x == door_pos[1] + 1:
                can_open = 1.0
            elif direction == 3 and agent_y == door_pos[0] + 1 and agent_x == door_pos[1]:
                can_open = 1.0

        return np.array(
            [
                agent_y,
                agent_x,
                goal_y,
                goal_x,
                direction,
                has_key,
                key_y,
                key_x,
                door_y,
                door_x,
                can_pickup,
                can_open,
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
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        # One-time bonus flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        self.gave_turn_bonus = False
        # Repeat action tracking
        self.last_action = None
        self.repeat_count = 0

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
        """Check if facing toward current target."""
        base_env = self._get_base_env()
        pos = base_env.agent_pos
        direction = base_env.agent_dir
        target = self._get_current_target()

        dy = target[0] - pos[0]
        dx = target[1] - pos[1]

        if direction == 0 and dx > 0:
            return True  # facing right, target is right
        if direction == 1 and dy > 0:
            return True  # facing down, target is down
        if direction == 2 and dx < 0:
            return True  # facing left, target is left
        if direction == 3 and dy < 0:
            return True  # facing up, target is up
        return False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._distance_to_target()
        # Reset all flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        self.gave_turn_bonus = False
        self.last_action = None
        self.repeat_count = 0
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

        # Key pickup bonus
        if has_key_now and not had_key_before:
            reward += REWARD_KEY_PICKUP
            self.previous_distance = self._distance_to_target()
            self.gave_turn_bonus = False
            return obs, reward, terminated, truncated, info

        # Door open bonus - check ALL doors
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos in base_env.door_positions:
                state_before = door_states_before.get(door_pos, 0)
                state_now = base_env.grid[door_pos[0], door_pos[1], 2]
                if state_before == 2 and state_now == 0:  # Was locked, now open
                    reward += REWARD_DOOR_OPEN
                    self.previous_distance = self._distance_to_target()
                    self.gave_turn_bonus = False
                    return obs, reward, terminated, truncated, info

        # Distance reward (only reached if no key pickup or door open)
        current_distance = self._distance_to_target()
        if current_distance < self.previous_distance:
            reward += REWARD_CLOSER
        elif current_distance > self.previous_distance:
            reward += PENALTY_FURTHER * 0
        self.previous_distance = current_distance

        # One-time turn bonus when facing target
        if action in [0, 1] and not self.gave_turn_bonus:
            if self._facing_target():
                reward += REWARD_TURN
                self.gave_turn_bonus = True

        # One-time hint when can pickup key
        if not has_key_now and not self.gave_pickup_hint:
            # Check can_pickup from observation (index 10)
            flat_obs = self._get_flat_obs()
            if flat_obs is not None and flat_obs[10] == 1.0:
                reward += 2.0
                self.gave_pickup_hint = True

        # One-time hint when can open door
        if has_key_now and not self.gave_door_hint:
            flat_obs = self._get_flat_obs()
            if flat_obs is not None and flat_obs[11] == 1.0:
                reward += 2.0
                self.gave_door_hint = True

        # Penalty for repeating non-moving actions
        if action == self.last_action and not moved:
            self.repeat_count += 1
            if self.repeat_count > 3:
                reward -= 0.2
        else:
            self.repeat_count = 0
        self.last_action = action

        return obs, reward, terminated, truncated, info

    def _get_flat_obs(self):
        """Get the flattened observation to check can_pickup/can_open."""
        base_env = self._get_base_env()
        agent_pos = base_env.agent_pos
        direction = base_env.agent_dir

        # Check can_pickup
        can_pickup = 0.0
        if (
            hasattr(base_env, "key_positions")
            and base_env.key_positions
            and base_env.carrying is None
        ):
            key_pos = list(base_env.key_positions.keys())[0]
            if direction == 0 and agent_pos[0] == key_pos[0] and agent_pos[1] == key_pos[1] - 1:
                can_pickup = 1.0
            elif direction == 1 and agent_pos[0] == key_pos[0] - 1 and agent_pos[1] == key_pos[1]:
                can_pickup = 1.0
            elif direction == 2 and agent_pos[0] == key_pos[0] and agent_pos[1] == key_pos[1] + 1:
                can_pickup = 1.0
            elif direction == 3 and agent_pos[0] == key_pos[0] + 1 and agent_pos[1] == key_pos[1]:
                can_pickup = 1.0

        # Check can_open
        can_open = 0.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            if direction == 0 and agent_pos[0] == door_pos[0] and agent_pos[1] == door_pos[1] - 1:
                can_open = 1.0
            elif (
                direction == 1 and agent_pos[0] == door_pos[0] - 1 and agent_pos[1] == door_pos[1]
            ):
                can_open = 1.0
            elif (
                direction == 2 and agent_pos[0] == door_pos[0] and agent_pos[1] == door_pos[1] + 1
            ):
                can_open = 1.0
            elif (
                direction == 3 and agent_pos[0] == door_pos[0] + 1 and agent_pos[1] == door_pos[1]
            ):
                can_open = 1.0

        # Return as array with indices 10 and 11
        return [0] * 10 + [can_pickup, can_open]

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
