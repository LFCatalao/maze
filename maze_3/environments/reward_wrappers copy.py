"""
Simplified Reward Wrapper for RL Experiments
Three-phase: Key -> Door -> Goal
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

REWARD_GOAL = 100.0
REWARD_CLOSER = 1.0
PENALTY_FURTHER = -0.5
PENALTY_STEP = -0.1
REWARD_TURN = 0.5
REWARD_KEY_PICKUP = 25.0
REWARD_DOOR_OPEN = 25.0
REWARD_CAN_INTERACT = 2.0  # Bonus when in position to pickup/open


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_adjacent_positions(pos):
    """Return 4 adjacent positions: above, below, left, right"""
    return [
        (pos[0] - 1, pos[1]),  # above
        (pos[0] + 1, pos[1]),  # below
        (pos[0], pos[1] - 1),  # left
        (pos[0], pos[1] + 1),  # right
    ]


def can_interact_with(agent_pos, agent_dir, target_pos):
    """Check if agent is adjacent to target and facing it."""
    # Direction: 0=right, 1=down, 2=left, 3=up
    if agent_dir == 0 and agent_pos[0] == target_pos[0] and agent_pos[1] == target_pos[1] - 1:
        return True  # Target is to the right, facing right
    if agent_dir == 1 and agent_pos[0] == target_pos[0] - 1 and agent_pos[1] == target_pos[1]:
        return True  # Target is below, facing down
    if agent_dir == 2 and agent_pos[0] == target_pos[0] and agent_pos[1] == target_pos[1] + 1:
        return True  # Target is to the left, facing left
    if agent_dir == 3 and agent_pos[0] == target_pos[0] + 1 and agent_pos[1] == target_pos[1]:
        return True  # Target is above, facing up
    return False


# def min_distance_to_adjacent(agent_pos, target_pos):
#     """Minimum distance to any cell adjacent to target."""
#     adjacent = get_adjacent_positions(target_pos)
#     distances = [abs(agent_pos[0] - adj[0]) + abs(agent_pos[1] - adj[1]) for adj in adjacent]
#     return min(distances)


# =============================================================================
# OBSERVATION WRAPPER
# =============================================================================


class SimpleObs(gym.ObservationWrapper):
    """
    Observation: [agent_y, agent_x, goal_y, goal_x, direction, has_key,
                  key_y, key_x, door_y, door_x, can_pickup, can_open_door]
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 20, (12,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_pos = base_env.agent_pos
        agent_y, agent_x = agent_pos
        goal_y, goal_x = base_env.goal_pos
        direction = base_env.agent_dir

        has_key = 1.0 if base_env.carrying is not None else 0.0

        # Key position and can_pickup
        key_y, key_x = -1.0, -1.0
        can_pickup = 0.0
        if (
            hasattr(base_env, "key_positions")
            and base_env.key_positions
            and base_env.carrying is None
        ):
            key_pos = list(base_env.key_positions.keys())[0]
            key_y, key_x = float(key_pos[0]), float(key_pos[1])
            if can_interact_with(agent_pos, direction, key_pos):
                can_pickup = 1.0

        # Door position and can_open
        door_y, door_x = -1.0, -1.0
        can_open = 0.0
        door_is_open = 0.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_y, door_x = float(door_pos[0]), float(door_pos[1])
            door_info = base_env.door_positions[door_pos]
            door_state = door_info[1] if isinstance(door_info, tuple) else door_info
            door_is_open = 1.0 if door_state == 0 else 0.0
            if can_interact_with(agent_pos, direction, door_pos):
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
    Three-phase reward:
    - Phase 1: Get the key
    - Phase 2: Open the door
    - Phase 3: Reach the goal
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        self.can_get_turn_bonus = True
        self.door_was_open = False
        self.gave_pickup_hint = False
        self.gave_door_hint = False

    def _get_base_env(self):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        return base_env

    def _get_phase(self):
        """Determine current phase: 'key', 'door', or 'goal'"""
        base_env = self._get_base_env()

        # Check if door is locked (need to check grid, not just door_positions)
        door_locked = False
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_state = base_env.grid[door_pos[0], door_pos[1], 2]
            door_locked = door_state == 2

        # Phase 1: Need key (only if door is still locked AND we don't have key)
        if door_locked and base_env.carrying is None:
            # Check if key still exists in the world
            if hasattr(base_env, "key_positions") and base_env.key_positions:
                return "key"

        # Phase 2: Have key (or no key needed), door still locked
        if door_locked:
            return "door"

        # Phase 3: Door is open (or no door), go to goal
        return "goal"

    def _get_current_target(self):
        """Return current target position based on phase."""
        base_env = self._get_base_env()
        phase = self._get_phase()

        if phase == "key":
            return list(base_env.key_positions.keys())[0]
        elif phase == "door":
            return list(base_env.door_positions.keys())[0]
        else:
            return base_env.goal_pos

    def _distance_to_target(self):
        """Distance to current target."""
        base_env = self._get_base_env()
        pos = base_env.agent_pos
        phase = self._get_phase()

        if phase == "key":
            key_pos = list(base_env.key_positions.keys())[0]
            # Distance to nearest adjacent cell
            distances = [
                abs(pos[0] - (key_pos[0] - 1)) + abs(pos[1] - key_pos[1]),
                abs(pos[0] - (key_pos[0] + 1)) + abs(pos[1] - key_pos[1]),
                abs(pos[0] - key_pos[0]) + abs(pos[1] - (key_pos[1] - 1)),
                abs(pos[0] - key_pos[0]) + abs(pos[1] - (key_pos[1] + 1)),
            ]
            return min(distances)

        elif phase == "door":
            door_pos = list(base_env.door_positions.keys())[0]
            # Distance to nearest adjacent cell
            distances = [
                abs(pos[0] - (door_pos[0] - 1)) + abs(pos[1] - door_pos[1]),
                abs(pos[0] - (door_pos[0] + 1)) + abs(pos[1] - door_pos[1]),
                abs(pos[0] - door_pos[0]) + abs(pos[1] - (door_pos[1] - 1)),
                abs(pos[0] - door_pos[0]) + abs(pos[1] - (door_pos[1] + 1)),
            ]
            return min(distances)

        else:  # goal phase
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
            return True
        if direction == 1 and dy > 0:
            return True
        if direction == 2 and dx < 0:
            return True
        if direction == 3 and dy < 0:
            return True
        return False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._distance_to_target()
        self.can_get_turn_bonus = True
        self.door_was_open = False
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        return obs, info

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)
        base_env = self._get_base_env()

        had_key = base_env.carrying is not None
        old_phase = self._get_phase()

        # Check door state before
        door_open_before = False
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_info = base_env.door_positions[door_pos]
            door_state = door_info[1] if isinstance(door_info, tuple) else door_info
            door_open_before = door_state == 0

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        moved = old_pos != new_pos
        has_key = base_env.carrying is not None
        new_phase = self._get_phase()

        # Goal reached
        if base_reward > 0:
            return obs, REWARD_GOAL, terminated, truncated, info

        reward = 0.0

        # Key pickup bonus
        if has_key and not had_key:
            reward += REWARD_KEY_PICKUP
            self.previous_distance = self._distance_to_target()
            return obs, reward + PENALTY_STEP, terminated, truncated, info

        # Door open bonus
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_info = base_env.door_positions[door_pos]
            door_state = door_info[1] if isinstance(door_info, tuple) else door_info
            door_open_now = door_state == 0
            if door_open_now and not door_open_before:
                reward += REWARD_DOOR_OPEN
                self.previous_distance = self._distance_to_target()
                return obs, reward + PENALTY_STEP, terminated, truncated, info

        # Distance reward
        current_distance = self._distance_to_target()
        if current_distance < self.previous_distance:
            reward += REWARD_CLOSER
        elif current_distance > self.previous_distance:
            reward += PENALTY_FURTHER
        self.previous_distance = current_distance

        # Turn bonus
        if action in [0, 1] and self.can_get_turn_bonus:
            if self._facing_target():
                reward += REWARD_TURN
                self.can_get_turn_bonus = False

        if moved:
            self.can_get_turn_bonus = True

        # Interaction hint bonus (only once per phase!)
        target = self._get_current_target()
        phase = self._get_phase()

        if can_interact_with(new_pos, base_env.agent_dir, target):
            if phase == "key" and not self.gave_pickup_hint:
                reward += REWARD_CAN_INTERACT
                self.gave_pickup_hint = True
            elif phase == "door" and not self.gave_door_hint:
                reward += REWARD_CAN_INTERACT
                self.gave_door_hint = True

        # If can open door but didn't toggle, small penalty
        if phase == "door" and can_interact_with(new_pos, base_env.agent_dir, target):
            if action != 4:  # Not TOGGLE
                reward -= 0.5  # Penalty for not toggling when you should

        # Step penalty
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
    if reward_type not in REWARD_WRAPPERS:
        raise ValueError(f"Unknown reward type: {reward_type}")
    return REWARD_WRAPPERS[reward_type](env)
