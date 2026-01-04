"""
Environment Configurations for RL Comparison Study

Defines preset configurations for different experimental conditions.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


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
    defined_doors: Optional[List[Dict]] = None
    goal_in_locked_room: bool = False
    enable_key_chain: bool = False

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

# E3_FIXED: Fixed initial position with 1 fixed door with key position fixed and fixed exit in the room blocked by the door
E3_FIXED_DOOR_KEY = EnvConfig(
    name="E3_fixed_door_key",
    description="Fixed agent, 1 fixed door, fixed key, fixed exit behind door",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=(3, 3), # Top Left Room
    defined_doors=[
        {'door_idx': 0, 'key_pos': (15, 15)} # Door 0 (Top Left), Key in Bot Right
    ],
    max_steps=400,
)

# E4: Fixed positions, multiple doors
E4_RANDOM_AGENT_KEYDOOR = EnvConfig(
    name="E4_random_agent_keydoor",
    description="Random agent, fixed goal, locked door",
    fixed_agent_pos=None,
    fixed_goal_pos=(3, 3),
    num_doors=1,
    include_key=True,
    locked_door=True,
    randomize_doors=False,
    max_steps=400,
)

# E5: Random agent position, fixed goal, one locked door
E5_RANDOM_AGENT = EnvConfig(
    name="E5_random_agent",
    description="Random agent start, fixed goal, one locked door",
    fixed_agent_pos=None,  # Random
    fixed_goal_pos=(3, 3),
    num_doors=1,
    include_key=True,
    locked_door=True,
    randomize_doors=False,
    max_steps=400,
)

# E6: Random agent AND random goal/door (same room)
E6_FULLY_RANDOM = EnvConfig(
    name="E6_fully_random",
    description="Random agent, random goal with door in same room",
    fixed_agent_pos=None,
    fixed_goal_pos=None,  # Will trigger same-room logic
    num_doors=1,
    include_key=True,
    locked_door=True,
    randomize_doors=True,  # Door placed at entrance to goal's room
    max_steps=500,
)

# E4_RANDOM: Fixed initial position with with a random exit in a room blocked by a random door and the key in one other random room different from exit
E4_RANDOM_LOCKED = EnvConfig(
    name="E4_random_locked",
    description="Fixed agent, random exit behind random door, random key",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=None,
    num_doors=1,
    include_key=True,
    locked_door=True,
    randomize_doors=True,
    goal_in_locked_room=True,
    max_steps=400,
)

# E5: Custom doors configuration
E5_CUSTOM_DOORS = EnvConfig(
    name="E5_custom_doors",
    description="Custom door and key placement",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=(3, 3), # Top Left
    defined_doors=[
        # Door 0: Top Left (Yellow)
        {'door_idx': 0, 'key_pos': (15, 15)}, # Key fixed in Bot Right
        # Door 2: Bot Left (Blue)
        {'door_idx': 2, 'key_pos': None},     # Key Random
    ],
    max_steps=500,
)

# E5_MULTI_FIXED: Fixed initial position with 4 fixed doors and the exit in a room blocked by one of them with the respective keys in the other rooms
E5_MULTI_DOOR_FIXED = EnvConfig(
    name="E5_multi_door_fixed",
    description="Fixed agent, 4 fixed doors, exit behind one, keys in others",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=(3, 3), # Top Left (Behind Door 0)
    defined_doors=[
        {'door_idx': 0, 'key_pos': (3, 15)}, # Top Left Door -> Key in Bot Left (Room 2, no door)
        {'door_idx': 1, 'key_pos': (15, 3)}, # Mid Left Door -> Key in Top Right (Room 3)
        {'door_idx': 3, 'key_pos': (9, 15)}, # Top Right Door -> Key in Mid Right (Room 4)
        {'door_idx': 4, 'key_pos': (9, 3)},  # Mid Right Door -> Key in Mid Left (Room 1)
    ],
    max_steps=500,
)

# E6_MULTI_RANDOM: Fixed initial position with a random exit blocked by a random door and the keys in random rooms
E6_MULTI_DOOR_RANDOM = EnvConfig(
    name="E6_multi_door_random",
    description="Fixed agent, random exit behind random door, keys in random rooms",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=None,
    num_doors=4,
    include_key=True,
    locked_door=True,
    randomize_doors=True,
    goal_in_locked_room=True,
    enable_key_chain=True,
    max_steps=500,
)

# E4_RANDOM: Fixed initial position with with a random exit in a room blocked by a random door and the key in one other random room different from exit
E4_RANDOM_LOCKED = EnvConfig(
    name="E4_random_locked",
    description="Fixed agent, random exit behind random door, random key",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=None,
    num_doors=1,
    include_key=True,
    locked_door=True,
    randomize_doors=True,
    goal_in_locked_room=True,
    max_steps=400,
)

# E5: Custom doors configuration
E5_CUSTOM_DOORS = EnvConfig(
    name="E5_custom_doors",
    description="Custom door and key placement",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=(3, 3), # Top Left
    defined_doors=[
        # Door 0: Top Left (Yellow)
        {'door_idx': 0, 'key_pos': (15, 15)}, # Key fixed in Bot Right
        # Door 2: Bot Left (Blue)
        {'door_idx': 2, 'key_pos': None},     # Key Random
    ],
    max_steps=500,
)

# E5_MULTI_FIXED: Fixed initial position with 4 fixed doors and the exit in a room blocked by one of them with the respective keys in the other rooms
E5_MULTI_DOOR_FIXED = EnvConfig(
    name="E5_multi_door_fixed",
    description="Fixed agent, 4 fixed doors, exit behind one, keys in others",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=(3, 3), # Top Left (Behind Door 0)
    defined_doors=[
        {'door_idx': 0, 'key_pos': (3, 15)}, # Top Left Door -> Key in Bot Left (Room 2, no door)
        {'door_idx': 1, 'key_pos': (15, 3)}, # Mid Left Door -> Key in Top Right (Room 3)
        {'door_idx': 3, 'key_pos': (9, 15)}, # Top Right Door -> Key in Mid Right (Room 4)
        {'door_idx': 4, 'key_pos': (9, 3)},  # Mid Right Door -> Key in Mid Left (Room 1)
    ],
    max_steps=500,
)

# E6_MULTI_RANDOM: Fixed initial position with a random exit blocked by a random door and the keys in random rooms
E6_MULTI_DOOR_RANDOM = EnvConfig(
    name="E6_multi_door_random",
    description="Fixed agent, random exit behind random door, keys in random rooms",
    fixed_agent_pos=(9, 9),
    fixed_goal_pos=None,
    num_doors=4,
    include_key=True,
    locked_door=True,
    randomize_doors=True,
    goal_in_locked_room=True,
    enable_key_chain=True,
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
    "E3_FIXED": E3_FIXED_DOOR_KEY,
    "E4": E4_RANDOM_AGENT_KEYDOOR,
    "E4_RANDOM": E4_RANDOM_LOCKED,
    "E5": E5_CUSTOM_DOORS,
    "E5_FIXED": E5_MULTI_DOOR_FIXED,
    "E6_RANDOM": E6_MULTI_DOOR_RANDOM,
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
