"""
Environments package for RL Comparison Study
"""

from .base_env import LockedRoomEnv, Objects, Colors, Actions, COLOR_NAMES, COLOR_MAP
from .configs import (
    EnvConfig,
    E1_FIXED_SIMPLE,
    E2_RANDOM_EXIT,
    E3_ONE_DOOR,
    E4_MULTI_DOOR,
    get_config,
    list_configs,
    ALL_ENV_CONFIGS,
)
from .reward_wrappers import (
    SimpleRewardWrapper,
    SimpleObs,
    apply_reward_wrapper,
    REWARD_WRAPPERS,
    REWARD_GOAL,
    REWARD_CLOSER,
    PENALTY_FURTHER,
    PENALTY_STEP,
    REWARD_TURN,
    REWARD_KEY_PICKUP,
    # REWARD_DOOR_OPEN,
    # REWARD_CAN_INTERACT,
)

__all__ = [
    # Environment
    "LockedRoomEnv",
    # Configs
    "EnvConfig",
    "get_config",
    "list_configs",
    "ALL_ENV_CONFIGS",
    # Reward wrappers
    "SimpleRewardWrapper",
    "SimpleObs",
    "apply_reward_wrapper",
    "REWARD_WRAPPERS",
    "REWARD_GOAL",
    "REWARD_CLOSER",
    "PENALTY_FURTHER",
    "PENALTY_STEP",
    "REWARD_TURN",
    "REWARD_KEY_PICKUP",
    # "REWARD_DOOR_OPEN",
    # "REWARD_CAN_INTERACT",
]
