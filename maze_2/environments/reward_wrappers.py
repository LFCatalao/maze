"""
Simplified Reward Wrapper for RL Experiments

Single reward function with adjustable hyperparameters.
"""

import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# HYPERPARAMETERS (adjust these for experiments)
# =============================================================================

REWARD_GOAL = 100.0  # Reward for reaching the goal
REWARD_CLOSER = 1.0  # Reward for getting closer to target
PENALTY_FURTHER = -0.5  # Penalty for moving away from target
PENALTY_STEP = -0.1  # Penalty per step (encourages efficiency)
REWARD_ROOM = 10.0  # Reward for entering goal room
REWARD_TURN = 0.5  # Reward for turning toward target


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


def is_facing_target(agent_pos, agent_dir, target_pos):
    """Check if agent is facing toward the target."""
    dy = target_pos[0] - agent_pos[0]
    dx = target_pos[1] - agent_pos[1]

    # Direction: 0=right, 1=down, 2=left, 3=up
    if agent_dir == 0 and dx > 0:
        return True
    if agent_dir == 2 and dx < 0:
        return True
    if agent_dir == 3 and dy < 0:
        return True
    if agent_dir == 1 and dy > 0:
        return True

    return False


# =============================================================================
# OBSERVATION WRAPPER
# =============================================================================


class SimpleObs(gym.ObservationWrapper):
    """
    Simplified observation: just position and direction.

    Observation: [agent_y, agent_x, goal_y, goal_x, direction]

    This is much easier for MLP to learn than the full grid (1094 values).
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 20, (5,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        return np.array(
            [
                base_env.agent_pos[0],
                base_env.agent_pos[1],
                base_env.goal_pos[0],
                base_env.goal_pos[1],
                base_env.agent_dir,
            ],
            dtype=np.float32,
        )


# =============================================================================
# REWARD WRAPPER
# =============================================================================


class SimpleRewardWrapper(gym.RewardWrapper):
    """
    Simple reward based on:
    - Distance to goal (closer = good)
    - Step penalty (fewer steps = good)
    - Room bonus (entering goal room)
    - Turn bonus (facing target)
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        self.in_goal_room = False
        self.can_get_turn_bonus = True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        goal = self.env.goal_pos
        agent = self.env.agent_pos

        self.previous_distance = manhattan_distance(agent, goal)
        self.in_goal_room = get_room(agent) == get_room(goal)
        self.can_get_turn_bonus = True

        return obs, info

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        goal_pos = self.env.goal_pos
        moved = old_pos != new_pos

        # Goal reached
        if base_reward > 0:
            return obs, REWARD_GOAL, terminated, truncated, info

        reward = 0.0

        # 1. Distance reward
        current_distance = manhattan_distance(new_pos, goal_pos)

        if current_distance < self.previous_distance:
            reward += REWARD_CLOSER
        elif current_distance > self.previous_distance:
            reward += PENALTY_FURTHER

        self.previous_distance = current_distance

        # 2. Turn bonus (once per move)
        if action in [0, 1] and self.can_get_turn_bonus:
            if is_facing_target(new_pos, self.env.agent_dir, goal_pos):
                reward += REWARD_TURN
                self.can_get_turn_bonus = False

        if moved:
            self.can_get_turn_bonus = True

        # 3. Room bonus
        current_room = get_room(new_pos)
        goal_room = get_room(goal_pos)

        if current_room == goal_room and not self.in_goal_room:
            reward += REWARD_ROOM
            self.in_goal_room = True
        elif current_room != goal_room:
            self.in_goal_room = False

        # 4. Step penalty
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
