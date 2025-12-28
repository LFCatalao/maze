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

REWARD_GOAL = 100.0  # Reward for reaching the goal
REWARD_CLOSER = 1.0  # Reward for getting closer to target
PENALTY_FURTHER = -0.5  # Penalty for moving away from target
PENALTY_STEP = -0.1  # Penalty per step (encourages efficiency)
REWARD_ROOM = 10.0  # Reward for entering goal room
REWARD_TURN = 0.5  # Reward for turning toward target
REWARD_KEY_PICKUP = 20.0  # Reward for picking up key


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
    Two-phase reward:
    - Phase 1: Get the key (only key matters)
    - Phase 2: Get to goal (only goal matters)
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        self.can_get_turn_bonus = True

    def _get_base_env(self):
        """Get the unwrapped base environment."""
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        return base_env

    def _get_current_target(self):
        """Return current target position based on phase."""
        base_env = self._get_base_env()

        # Phase 1: No key yet -> target is key
        if (
            base_env.carrying is None
            and hasattr(base_env, "key_positions")
            and base_env.key_positions
        ):
            return list(base_env.key_positions.keys())[0]

        # Phase 2: Have key -> target is goal
        return base_env.goal_pos

    def _distance_to_target(self):
        """Distance to current target."""
        base_env = self._get_base_env()
        pos = base_env.agent_pos

        # Phase 1: Need key - go to adjacent cell
        if (
            base_env.carrying is None
            and hasattr(base_env, "key_positions")
            and base_env.key_positions
        ):
            key_pos = list(base_env.key_positions.keys())[0]
            # Distance to nearest adjacent cell (not the key itself)
            # Adjacent cells: (key_y-1, key_x), (key_y+1, key_x), (key_y, key_x-1), (key_y, key_x+1)
            distances = [
                abs(pos[0] - (key_pos[0] - 1)) + abs(pos[1] - key_pos[1]),  # above key
                abs(pos[0] - (key_pos[0] + 1)) + abs(pos[1] - key_pos[1]),  # below key
                abs(pos[0] - key_pos[0]) + abs(pos[1] - (key_pos[1] - 1)),  # left of key
                abs(pos[0] - key_pos[0]) + abs(pos[1] - (key_pos[1] + 1)),  # right of key
            ]
            return min(distances)

        # Phase 2: Have key - go to goal
        goal = base_env.goal_pos
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _facing_target(self):
        """Check if facing toward current target."""
        base_env = self._get_base_env()
        pos = base_env.agent_pos
        direction = base_env.agent_dir
        target = self._get_current_target()

        dy = target[0] - pos[0]
        dx = target[1] - pos[1]

        if direction == 0 and dx > 0:
            return True  # facing right, target right
        if direction == 2 and dx < 0:
            return True  # facing left, target left
        if direction == 3 and dy < 0:
            return True  # facing up, target up
        if direction == 1 and dy > 0:
            return True  # facing down, target down
        return False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._distance_to_target()
        self.can_get_turn_bonus = True
        return obs, info

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)
        base_env = self._get_base_env()
        had_key_before = base_env.carrying is not None

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        moved = old_pos != new_pos
        has_key_now = base_env.carrying is not None

        # Goal reached
        if base_reward > 0:
            return obs, REWARD_GOAL, terminated, truncated, info

        reward = 0.0

        # 1. Key pickup - BIG reward and reset distance tracking
        if has_key_now and not had_key_before:
            reward += REWARD_KEY_PICKUP
            self.previous_distance = self._distance_to_target()  # Now tracking goal
            return obs, reward + PENALTY_STEP, terminated, truncated, info

        # 2. Distance to current target (key or goal)
        current_distance = self._distance_to_target()
        if current_distance < self.previous_distance:
            reward += REWARD_CLOSER
        elif current_distance > self.previous_distance:
            reward += PENALTY_FURTHER
        self.previous_distance = current_distance

        # 3. Turn bonus
        if action in [0, 1] and self.can_get_turn_bonus:
            if self._facing_target():
                reward += REWARD_TURN
                self.can_get_turn_bonus = False

        if moved:
            self.can_get_turn_bonus = True

        # 4. Hint for being in position to pickup key
        if not has_key_now and hasattr(base_env, "key_positions") and base_env.key_positions:
            key_pos = list(base_env.key_positions.keys())[0]
            direction = base_env.agent_dir

            # Check if can pickup (adjacent and facing)
            can_pickup = False
            if direction == 0 and new_pos[0] == key_pos[0] and new_pos[1] == key_pos[1] - 1:
                can_pickup = True
            elif direction == 1 and new_pos[0] == key_pos[0] - 1 and new_pos[1] == key_pos[1]:
                can_pickup = True
            elif direction == 2 and new_pos[0] == key_pos[0] and new_pos[1] == key_pos[1] + 1:
                can_pickup = True
            elif direction == 3 and new_pos[0] == key_pos[0] + 1 and new_pos[1] == key_pos[1]:
                can_pickup = True

            if can_pickup:
                reward += 2.0  # Strong hint: you can pickup now!

        # 5. Step penalty
        reward += PENALTY_STEP

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
