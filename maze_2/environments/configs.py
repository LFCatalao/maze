"""
Environment Configurations for RL Comparison Study

Defines preset configurations for different experimental conditions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnvConfig:
    """Environment configuration preset"""

    name: str
    description: str

    # Grid settings
    size: int = 19
    max_steps: int = 500

    # Position settings
    fixed_agent_pos: Optional[Tuple[int, int]] = None  # (y, x) or None for random
    fixed_goal_pos: Optional[Tuple[int, int]] = None  # (y, x) or None for random

    # Door/key settings
    num_doors: int = 0
    include_key: bool = False
    locked_door: bool = False
    randomize_doors: bool = False

    # Observation settings
    observation_mode: str = "full_map"
    agent_view_size: int = 7


# =============================================================================
# ENVIRONMENT PRESETS
# =============================================================================

# E1: Fixed start, fixed exit, no doors (simplest)
E1_FIXED_SIMPLE = EnvConfig(
    name="E1_fixed_simple",
    description="Fixed start, fixed exit, no obstacles",
    fixed_agent_pos=(9, 9),  # Center of corridor
    fixed_goal_pos=(3, 3),  # Top-left room
    num_doors=0,
    include_key=False,
    locked_door=False,
    max_steps=200,
)

# E2: Fixed start, random exit, no doors
E2_RANDOM_EXIT = EnvConfig(
    name="E2_random_exit",
    description="Fixed start, random exit position",
    fixed_agent_pos=(9, 9),  # Center of corridor
    fixed_goal_pos=None,  # Random goal
    num_doors=0,
    include_key=False,
    locked_door=False,
    max_steps=300,
)

# E3: Fixed positions, one locked door
E3_ONE_DOOR = EnvConfig(
    name="E3_one_door",
    description="Fixed positions with one locked door requiring key",
    fixed_agent_pos=(15, 9),  # Bottom of corridor
    fixed_goal_pos=(3, 3),  # Top-left room (behind door)
    num_doors=1,
    include_key=True,
    locked_door=True,
    randomize_doors=False,
    max_steps=400,
)

# E4: Fixed positions, multiple doors
E4_MULTI_DOOR = EnvConfig(
    name="E4_multi_door",
    description="Fixed positions with multiple locked doors",
    fixed_agent_pos=(15, 9),  # Bottom of corridor
    fixed_goal_pos=(3, 3),  # Top-left room
    num_doors=4,
    include_key=True,
    locked_door=True,
    randomize_doors=False,
    max_steps=500,
)

# =============================================================================
# OBSERVATION MODE VARIANTS
# =============================================================================


def with_observation_mode(config: EnvConfig, obs_mode: str) -> EnvConfig:
    """Create a copy of config with different observation mode"""
    import copy

    new_config = copy.deepcopy(config)
    new_config.observation_mode = obs_mode
    new_config.name = f"{config.name}_{obs_mode}"
    return new_config


# =============================================================================
# ALL CONFIGURATIONS
# =============================================================================

ALL_ENV_CONFIGS = {
    "E1": E1_FIXED_SIMPLE,
    "E2": E2_RANDOM_EXIT,
    "E3": E3_ONE_DOOR,
    "E4": E4_MULTI_DOOR,
}


def get_config(config_id: str, observation_mode: str = "full_map") -> EnvConfig:
    """Get a configuration by ID with optional observation mode override"""
    if config_id not in ALL_ENV_CONFIGS:
        raise ValueError(f"Unknown config: {config_id}. Available: {list(ALL_ENV_CONFIGS.keys())}")

    config = ALL_ENV_CONFIGS[config_id]
    if observation_mode != "full_map":
        config = with_observation_mode(config, observation_mode)

    return config


def list_configs():
    """Print all available configurations"""
    print("Available Environment Configurations:")
    print("=" * 60)
    for config_id, config in ALL_ENV_CONFIGS.items():
        print(f"\n{config_id}: {config.name}")
        print(f"  {config.description}")
        print(
            f"  Doors: {config.num_doors}, Key: {config.include_key}, Locked: {config.locked_door}"
        )
        print(f"  Max steps: {config.max_steps}")


if __name__ == "__main__":
    list_configs()
