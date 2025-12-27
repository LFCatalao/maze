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
from ...maze_3.reward_wrappers import (
    SparseRewardWrapper,
    DistanceRewardWrapper,
    StepPenaltyRewardWrapper,
    CombinedRewardWrapper,
    apply_reward_wrapper,
    list_reward_types,
    REWARD_WRAPPERS,
)

__all__ = [
    # Environment
    "LockedRoomEnv",
    "Objects",
    "Colors",
    "Actions",
    "COLOR_NAMES",
    "COLOR_MAP",
    # Configs
    "EnvConfig",
    "E1_FIXED_SIMPLE",
    "E2_RANDOM_EXIT",
    "E3_ONE_DOOR",
    "E4_MULTI_DOOR",
    "get_config",
    "list_configs",
    "ALL_ENV_CONFIGS",
    # Rewards
    "SparseRewardWrapper",
    "DistanceRewardWrapper",
    "StepPenaltyRewardWrapper",
    "CombinedRewardWrapper",
    "apply_reward_wrapper",
    "list_reward_types",
    "REWARD_WRAPPERS",
]
