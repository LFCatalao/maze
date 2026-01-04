"""
Refactored Locked Room Environment

Key changes from original:
- Support for fixed agent/goal positions (for reproducible experiments)
- Cleaner separation of concerns
- Better documentation
"""

import numpy as np
import pygame
from enum import IntEnum
from typing import Dict, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces


class Objects(IntEnum):
    """Object types in the environment"""

    EMPTY = 0
    WALL = 1
    DOOR = 2
    KEY = 3
    GOAL = 4
    AGENT = 5


class Colors(IntEnum):
    """Color indices"""

    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5


class Actions(IntEnum):
    """Available actions"""

    LEFT = 0  # Turn left
    RIGHT = 1  # Turn right
    FORWARD = 2  # Move forward
    PICKUP = 3  # Pick up object
    TOGGLE = 4  # Toggle/activate object (open door)


# Color mapping for pygame rendering
COLOR_MAP = {
    Colors.RED: (255, 0, 0),
    Colors.GREEN: (0, 255, 0),
    Colors.BLUE: (0, 0, 255),
    Colors.PURPLE: (128, 0, 128),
    Colors.YELLOW: (255, 255, 0),
    Colors.GREY: (128, 128, 128),
}

COLOR_NAMES = {
    Colors.RED: "red",
    Colors.GREEN: "green",
    Colors.BLUE: "blue",
    Colors.PURPLE: "purple",
    Colors.YELLOW: "yellow",
    Colors.GREY: "grey",
}

# Maps each room to its door position (y, x) and door color
ROOM_DOOR_MAP = {
    "left_top": (3, 7, Colors.YELLOW),
    "left_middle": (9, 7, Colors.GREEN),
    "left_bottom": (15, 7, Colors.BLUE),
    "right_top": (3, 11, Colors.PURPLE),
    "right_middle": (9, 11, Colors.GREY),
    "right_bottom": (15, 11, Colors.RED),
}


class LockedRoomEnv(gym.Env):
    """
    Locked Room Environment for RL Comparison Study

    A grid-based navigation environment where an agent must:
    1. Navigate through rooms
    2. Optionally collect keys
    3. Optionally unlock doors
    4. Reach a goal position

    Key Features:
    - Configurable fixed or random positions
    - Variable number of doors/keys
    - Multiple observation modes
    - Compatible with Gymnasium API
    """

    metadata = {"render_modes": ["human", "rgb_array", "console"], "render_fps": 10}

    def __init__(
        self,
        size: int = 19,
        observation_mode: str = "full_map",
        render_mode: Optional[str] = "human",
        max_steps: int = 1000,
        agent_view_size: int = 7,
        # Position configuration
        fixed_agent_pos: Optional[Tuple[int, int]] = None,
        fixed_goal_pos: Optional[Tuple[int, int]] = None,
        # Door/key configuration
        num_doors: int = 0,
        randomize_doors: bool = False,
        include_key: bool = False,
        locked_door: bool = False,
        defined_doors: Optional[List[Dict]] = None,
        goal_in_locked_room: bool = False,
        enable_key_chain: bool = False,
        # Training options
        verbose: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            size: Grid size (default 19x19)
            observation_mode: "full_map", "partial_view", or "agent_view"
            render_mode: "human", "rgb_array", "console", or None
            max_steps: Maximum steps per episode
            agent_view_size: Size of agent's view window
            fixed_agent_pos: (y, x) fixed starting position, or None for random
            fixed_goal_pos: (y, x) fixed goal position, or None for random
            num_doors: Number of doors (0-6)
            randomize_doors: Whether to randomize door positions
            include_key: Whether to include a key
            locked_door: Whether one door should be locked
            defined_doors: List of dicts {'pos': (y,x), 'color': int, 'key_pos': (y,x)|None}
            goal_in_locked_room: If True, random goal is placed in a room behind a locked door
            enable_key_chain: If True, generates a dependency chain of keys and doors
            verbose: Print action feedback (for debugging)
        """
        super().__init__()

        self.size = size
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.agent_view_size = agent_view_size
        self.fixed_agent_pos = fixed_agent_pos
        self.fixed_goal_pos = fixed_goal_pos
        self.num_doors = max(0, min(6, num_doors))
        self.randomize_doors = randomize_doors
        self.include_key = include_key
        self.locked_door = locked_door
        self.defined_doors = defined_doors
        self.goal_in_locked_room = goal_in_locked_room
        self.enable_key_chain = enable_key_chain
        self.verbose = verbose
        
        self.key_chain_plan = None

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Setup observation space based on mode
        self._setup_observation_space()

        # Pygame setup (lazy initialization)
        self.cell_size = 32
        self.window = None
        self.clock = None

        # Environment state (initialized in reset)
        self.grid = None
        self.agent_pos = None
        self.agent_dir = 0
        self.carrying = None
        self.step_count = 0
        self.door_positions = {}
        self.key_positions = {}
        self.goal_pos = None
        self.locked_door_color = None
        self.mission = ""

    def _setup_observation_space(self):
        """Setup observation space based on observation mode"""
        if self.observation_mode == "full_map":
            image_shape = (self.size, self.size, 3)
        elif self.observation_mode == "partial_view":
            view_size = 2 * self.agent_view_size + 1
            image_shape = (view_size, view_size, 3)
        else:  # agent_view
            image_shape = (self.agent_view_size, self.agent_view_size, 3)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
                "direction": spaces.Discrete(4),
                "carrying": spaces.Discrete(len(Colors) + 1),
            }
        )

    def _get_valid_empty_positions(self) -> List[Tuple[int, int]]:
        """Get list of valid empty positions for placing objects"""
        valid = []
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                if self.grid[y, x, 0] == Objects.EMPTY:
                    valid.append((y, x))
        return valid

    def _get_room_index(self, y: int, x: int) -> int:
        """
        Get room index for a given position.
        Rooms:
        0: Top-Left, 1: Mid-Left, 2: Bot-Left
        3: Top-Right, 4: Mid-Right, 5: Bot-Right
        -1: Corridor or Wall
        """
        if not (1 <= y < self.size - 1 and 1 <= x < self.size - 1):
            return -1
        
        # Corridor check (x=7, 11 are walls, 8-10 is corridor)
        if 7 <= x <= 11:
            return -1
            
        # Left side
        if x < 7:
            if y <= 5: return 0
            if y <= 11: return 1
            return 2
        
        # Right side
        if x > 11:
            if y <= 5: return 3
            if y <= 11: return 4
            return 5
            
        return -1

    def _get_random_pos_in_room(self, room_idx: int) -> Optional[Tuple[int, int]]:
        """Get a random empty position in the specified room"""
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
            return None
            
        y1, y2, x1, x2 = bounds[room_idx]
        
        # Find all empty spots in this room
        valid = []
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if self.grid[y, x, 0] == Objects.EMPTY:
                    valid.append((y, x))
                    
        if not valid:
            return None
            
        return valid[np.random.randint(len(valid))]

    def _generate_grid(self):
        """Generate the grid layout with walls and door openings"""
        # Initialize empty grid
        self.grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # Outer walls
        self.grid[0, :, 0] = Objects.WALL
        self.grid[-1, :, 0] = Objects.WALL
        self.grid[:, 0, 0] = Objects.WALL
        self.grid[:, -1, 0] = Objects.WALL

        # Interior walls creating rooms (corridor at x=8-10)
        for y in range(self.size):
            self.grid[y, 7, 0] = Objects.WALL
            self.grid[y, 11, 0] = Objects.WALL

        # Horizontal walls dividing rooms
        for x in range(1, 7):
            self.grid[6, x, 0] = Objects.WALL
            self.grid[12, x, 0] = Objects.WALL
        for x in range(12, 18):
            self.grid[6, x, 0] = Objects.WALL
            self.grid[12, x, 0] = Objects.WALL

        # Define door positions (openings in walls)
        self.all_door_positions = [
            (3, 7, Colors.YELLOW),  # Top-left room
            (9, 7, Colors.GREEN),  # Middle-left room
            (15, 7, Colors.BLUE),  # Bottom-left room
            (3, 11, Colors.PURPLE),  # Top-right room
            (9, 11, Colors.GREY),  # Middle-right room
            (15, 11, Colors.RED),  # Bottom-right room
        ]

        # Create openings at all door positions
        for y, x, _ in self.all_door_positions:
            self.grid[y, x, 0] = Objects.EMPTY

    def _place_doors(self):
        """Place doors according to configuration"""
        self.door_positions = {}
        self.keys_to_place = []  # List of (color, fixed_pos_or_None)

        if self.defined_doors is not None:
            # Use explicit configuration based on predefined door indices
            for door_conf in self.defined_doors:
                idx = door_conf.get('door_idx')
                
                if idx is None or not (0 <= idx < len(self.all_door_positions)):
                    if self.verbose:
                        print(f"Warning: Invalid door index {idx}. Must be 0-5.")
                    continue

                # Get predefined position and default color
                y, x, default_color = self.all_door_positions[idx]
                
                # Allow color override, otherwise use default
                color = door_conf.get('color', default_color)
                key_pos = door_conf.get('key_pos')
                
                # Place locked door
                self.grid[y, x, 0] = Objects.DOOR
                self.grid[y, x, 1] = color
                self.grid[y, x, 2] = 2  # Always locked
                self.door_positions[(y, x)] = (color, 2)
                
                # Add to keys to place
                self.keys_to_place.append((color, key_pos))
            return

        if self.enable_key_chain and self.num_doors > 0:
            # Generate key chain logic
            all_indices = list(range(len(self.all_door_positions)))
            # Ensure we have enough rooms for the chain + 1 open room
            if self.num_doors > len(all_indices) - 1:
                 # Fallback if too many doors requested
                 selected_indices = np.random.choice(all_indices, self.num_doors, replace=False)
            else:
                 selected_indices = np.random.choice(all_indices, self.num_doors, replace=False)
            
            open_indices = [i for i in all_indices if i not in selected_indices]
            
            # Shuffle selected indices to form a chain
            chain_indices = list(selected_indices)
            np.random.shuffle(chain_indices)
            
            # Start room (where the first key is) - must be an open room
            if open_indices:
                start_room_idx = np.random.choice(open_indices)
            else:
                start_room_idx = -1 # Corridor or fallback
            
            # Plan:
            # Key for chain_indices[0] -> in start_room_idx
            # Key for chain_indices[1] -> in chain_indices[0]
            # ...
            # Key for chain_indices[i] -> in chain_indices[i-1]
            # Goal -> in chain_indices[-1]
            
            self.key_chain_plan = {
                'goal_room': chain_indices[-1],
                'key_placements': []
            }
            
            # Place doors
            for idx in selected_indices:
                y, x, color = self.all_door_positions[idx]
                self.grid[y, x, 0] = Objects.DOOR
                self.grid[y, x, 1] = color
                self.grid[y, x, 2] = 2 # Locked
                self.door_positions[(y, x)] = (color, 2)
                
            # Define key placements
            # First key (opens first door in chain)
            first_door_idx = chain_indices[0]
            first_door_color = self.all_door_positions[first_door_idx][2]
            self.key_chain_plan['key_placements'].append((first_door_color, start_room_idx))
            
            # Subsequent keys
            for i in range(1, len(chain_indices)):
                target_door_idx = chain_indices[i]
                target_door_color = self.all_door_positions[target_door_idx][2]
                prev_room_idx = chain_indices[i-1]
                self.key_chain_plan['key_placements'].append((target_door_color, prev_room_idx))
                
            return

        if self.num_doors == 0:
            return

        if self.randomize_doors and self.goal_pos is not None:
            # Place door at entrance to goal's room
            goal_room = self._get_room(self.goal_pos)

            if goal_room in ROOM_DOOR_MAP:
                y, x, color = ROOM_DOOR_MAP[goal_room]
                state = 2 if self.locked_door else 1  # 2=locked, 1=closed
                self.grid[y, x, 0] = Objects.DOOR
                self.grid[y, x, 1] = color
                self.grid[y, x, 2] = state
                self.door_positions[(y, x)] = (color, state)
                if self.locked_door:
                    self.locked_door_color = color
            return

        # Original logic for non-randomized doors
        selected = self.all_door_positions[: self.num_doors]

        lock_idx = 0
        if self.locked_door and len(selected) > 0:
            self.locked_door_color = selected[lock_idx][2]
            self.keys_to_place.append((self.locked_door_color, None))

        for i, (y, x, color) in enumerate(selected):
            state = 2 if (self.locked_door and i == lock_idx) else 1
            self.grid[y, x, 0] = Objects.DOOR
            self.grid[y, x, 1] = color
            self.grid[y, x, 2] = state
            self.door_positions[(y, x)] = (color, state)

    def _get_room(self, pos):
        """Determine which room a position is in."""
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
        else:  # x > 11
            if y < 6:
                return "right_top"
            elif y < 12:
                return "right_middle"
            else:
                return "right_bottom"

    def _place_key(self):
        """Place key in accessible area (not behind any locked door)."""
        self.key_positions = {}

        if self.enable_key_chain and self.key_chain_plan:
            for color, room_idx in self.key_chain_plan['key_placements']:
                if room_idx == -1:
                    # Place in corridor
                    valid = self._get_valid_empty_positions()
                    corridor = [p for p in valid if 8 <= p[1] <= 10]
                    if corridor:
                        pos = corridor[np.random.randint(len(corridor))]
                    else:
                        pos = valid[np.random.randint(len(valid))]
                else:
                    pos = self._get_random_pos_in_room(room_idx)
                
                if pos:
                    self.grid[pos[0], pos[1], 0] = Objects.KEY
                    self.grid[pos[0], pos[1], 1] = color
                    self.key_positions[pos] = color
            return

        # Case 1: Robust placement (defined doors or legacy locked door)
        if self.keys_to_place:
            # Separate fixed and random keys
            fixed_keys = []
            random_keys = []
            for color, pos in self.keys_to_place:
                if pos is not None:
                    fixed_keys.append((color, pos))
                else:
                    random_keys.append(color)

            # Place fixed keys
            for color, pos in fixed_keys:
                self.grid[pos[0], pos[1], 0] = Objects.KEY
                self.grid[pos[0], pos[1], 1] = color
                self.key_positions[pos] = color

            if not random_keys:
                return

            # Identify available rooms
            available_rooms = set(range(6))
            
            # Remove goal room
            if self.goal_pos:
                goal_room = self._get_room_index(self.goal_pos[0], self.goal_pos[1])
                if goal_room in available_rooms:
                    available_rooms.remove(goal_room)
                
            # Remove rooms with fixed keys
            for _, pos in fixed_keys:
                r = self._get_room_index(pos[0], pos[1])
                if r in available_rooms:
                    available_rooms.remove(r)

            # Identify rooms without doors
            # Door positions mapping to rooms:
            # (3, 7) -> Room 0, (9, 7) -> Room 1, (15, 7) -> Room 2
            # (3, 11) -> Room 3, (9, 11) -> Room 4, (15, 11) -> Room 5
            door_map = {
                (3, 7): 0, (9, 7): 1, (15, 7): 2,
                (3, 11): 3, (9, 11): 4, (15, 11): 5
            }
            rooms_with_doors = set()
            for pos, room_idx in door_map.items():
                # Check if there is a door object at this position
                if self.grid[pos[0], pos[1], 0] == Objects.DOOR:
                    rooms_with_doors.add(room_idx)
            
            rooms_without_doors = available_rooms - rooms_with_doors
            
            # Constraint: One key in a room without a door
            # We need to pick one key to satisfy this, if possible
            keys_remaining = list(random_keys)
            
            if rooms_without_doors and keys_remaining:
                # Pick a room without a door
                room_idx = np.random.choice(list(rooms_without_doors))
                # Pick a key
                key_color = keys_remaining.pop(0)
                
                # Place key
                pos = self._get_random_pos_in_room(room_idx)
                if pos:
                    self.grid[pos[0], pos[1], 0] = Objects.KEY
                    self.grid[pos[0], pos[1], 1] = key_color
                    self.key_positions[pos] = key_color
                    available_rooms.remove(room_idx)
            
            # Place remaining keys in remaining available rooms
            # Shuffle available rooms
            avail_list = list(available_rooms)
            np.random.shuffle(avail_list)
            
            for key_color in keys_remaining:
                if not avail_list:
                    if self.verbose:
                        print(f"Warning: Not enough rooms to place key {COLOR_NAMES.get(key_color)}")
                    continue
                    
                room_idx = avail_list.pop(0)
                pos = self._get_random_pos_in_room(room_idx)
                if pos:
                    self.grid[pos[0], pos[1], 0] = Objects.KEY
                    self.grid[pos[0], pos[1], 1] = key_color
                    self.key_positions[pos] = key_color
            
            return

        # Case 2: Legacy configuration
        if not self.include_key:
            return

        # Get rooms that are NOT blocked by locked doors
        blocked_rooms = set()
        if self.locked_door and self.goal_pos is not None:
            goal_room = self._get_room(self.goal_pos)
            blocked_rooms.add(goal_room)

        # Valid key positions: corridor + unblocked rooms
        valid_key_positions = []
        for y, x in self._get_valid_empty_positions():
            room = self._get_room((y, x))
            if room == "corridor" or room not in blocked_rooms:
                valid_key_positions.append((y, x))

        if valid_key_positions:
            idx = np.random.randint(len(valid_key_positions))
            key_pos = valid_key_positions[idx]
            key_color = self.locked_door_color if self.locked_door else Colors.BLUE
            self.grid[key_pos[0], key_pos[1], 0] = Objects.KEY
            self.grid[key_pos[0], key_pos[1], 1] = key_color
            self.key_positions[key_pos] = key_color

    def _place_goal(self):
        """Place goal at fixed or random position"""
        if self.fixed_goal_pos is not None:
            goal_y, goal_x = self.fixed_goal_pos
        elif self.enable_key_chain and self.key_chain_plan:
            room_idx = self.key_chain_plan['goal_room']
            pos = self._get_random_pos_in_room(room_idx)
            if pos:
                goal_y, goal_x = pos
            else:
                # Fallback
                goal_y, goal_x = (3, 3)
        elif self.goal_in_locked_room:
            # Find rooms behind locked doors
            locked_rooms = []
            
            # Map door positions to room indices
            # (3, 7) -> Room 0, (9, 7) -> Room 1, (15, 7) -> Room 2
            # (3, 11) -> Room 3, (9, 11) -> Room 4, (15, 11) -> Room 5
            door_to_room = {
                (3, 7): 0, (9, 7): 1, (15, 7): 2,
                (3, 11): 3, (9, 11): 4, (15, 11): 5
            }
            
            for pos, (color, state) in self.door_positions.items():
                if state == 2: # Locked
                    if pos in door_to_room:
                        locked_rooms.append(door_to_room[pos])
            
            if locked_rooms:
                room_idx = locked_rooms[np.random.randint(len(locked_rooms))]
                pos = self._get_random_pos_in_room(room_idx)
                if pos:
                    goal_y, goal_x = pos
                else:
                    goal_y, goal_x = 3, 3 # Fallback
            else:
                # Fallback if no locked doors found
                if self.verbose:
                    print("Warning: goal_in_locked_room=True but no locked doors found.")
                goal_y, goal_x = 3, 3
        else:
            # Random position in one of the 6 rooms (not corridor)
            valid_positions = [
                (y, x)
                for y, x in self._get_valid_empty_positions()
                if self._get_room((y, x)) != "corridor"
            ]
            if valid_positions:
                idx = np.random.randint(len(valid_positions))
                goal_y, goal_x = valid_positions[idx]
            else:
                goal_y, goal_x = 3, 3  # Fallback

        self.grid[goal_y, goal_x, 0] = Objects.GOAL
        self.goal_pos = (goal_y, goal_x)

    def _place_agent(self):
        """Place agent at fixed or random position"""
        if self.fixed_agent_pos is not None:
            agent_y, agent_x = self.fixed_agent_pos
        else:
            # Random position anywhere that's empty (excluding goal room if doors exist)
            valid_positions = self._get_valid_empty_positions()

            # If there's a locked door, don't spawn agent in the goal's room
            if self.locked_door and self.goal_pos is not None:
                goal_room = self._get_room(self.goal_pos)
                valid_positions = [
                    (y, x) for y, x in valid_positions if self._get_room((y, x)) != goal_room
                ]

            if valid_positions:
                idx = np.random.randint(len(valid_positions))
                agent_y, agent_x = valid_positions[idx]
            else:
                # Fallback to corridor
                agent_y = np.random.randint(2, 17)
                agent_x = np.random.randint(8, 11)

        self.agent_pos = [agent_y, agent_x]
        self.agent_dir = np.random.randint(4)  # Random initial direction too

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self.step_count = 0
        self.carrying = None

        # Generate environment - ORDER MATTERS
        self._generate_grid()
        self._place_doors()
        self._place_goal()  # Place goal before keys to respect constraints
        self._place_key()
        self._place_agent()

        # Generate mission string
        self._generate_mission()

        return self._get_obs(), self._get_info()

    def _generate_mission(self):
        """Generate mission description string"""
        if self.num_doors == 0:
            self.mission = "Navigate to the goal"
        elif self.locked_door:
            color = COLOR_NAMES.get(self.locked_door_color, "colored")
            self.mission = f"Get the {color} key, unlock the door, and reach the goal"
        else:
            self.mission = "Open the door and reach the goal"

    def _get_obs(self) -> Dict:
        """Get current observation"""
        if self.observation_mode == "full_map":
            image = self.grid.copy()
        elif self.observation_mode == "partial_view":
            image = self._get_partial_view()
        else:
            image = self._get_agent_view()

        return {
            "image": image,
            "direction": self.agent_dir,
            "carrying": self.carrying if self.carrying is not None else len(Colors),
        }

    def _get_partial_view(self) -> np.ndarray:
        """Get partial view centered on agent"""
        half = self.agent_view_size
        y, x = self.agent_pos

        # Pad grid
        padded = np.zeros((self.size + 2 * half, self.size + 2 * half, 3), dtype=np.uint8)
        padded[half:-half, half:-half] = self.grid

        # Extract view
        return padded[y : y + 2 * half + 1, x : x + 2 * half + 1].copy()

    def _get_agent_view(self) -> np.ndarray:
        """Get first-person view"""
        view = np.zeros((self.agent_view_size, self.agent_view_size, 3), dtype=np.uint8)
        y, x = self.agent_pos

        # Direction vectors
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        right_vec = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        dy, dx = dir_vec[self.agent_dir]
        ry, rx = right_vec[self.agent_dir]

        for i in range(self.agent_view_size):
            for j in range(self.agent_view_size):
                offset = j - self.agent_view_size // 2
                wy = y + i * dy + offset * ry
                wx = x + i * dx + offset * rx

                if 0 <= wy < self.size and 0 <= wx < self.size:
                    view[i, j] = self.grid[wy, wx]

        return view

    def _get_info(self) -> Dict:
        """Get info dictionary"""
        return {
            "step_count": self.step_count,
            "agent_pos": self.agent_pos.copy(),
            "carrying": self.carrying,
            "mission": self.mission,
            "goal_pos": self.goal_pos,
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step"""
        self.step_count += 1

        # Execute action
        if action == Actions.LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == Actions.RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == Actions.FORWARD:
            self._move_forward()
        elif action == Actions.PICKUP:
            self._pickup()
        elif action == Actions.TOGGLE:
            self._toggle()

        # Check completion
        reward = 0.0
        terminated = False

        if tuple(self.agent_pos) == self.goal_pos:
            reward = 1.0
            terminated = True

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move_forward(self):
        """Move agent forward if possible"""
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy, dx = dir_vec[self.agent_dir]
        new_y, new_x = self.agent_pos[0] + dy, self.agent_pos[1] + dx

        if not (0 <= new_y < self.size and 0 <= new_x < self.size):
            return

        obj = self.grid[new_y, new_x, 0]

        if obj == Objects.WALL:
            return
        if obj == Objects.DOOR and self.grid[new_y, new_x, 2] != 0:
            return  # Door is closed/locked

        self.agent_pos = [new_y, new_x]

    def _pickup(self):
        """Pick up object in front"""
        if self.carrying is not None:
            return

        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy, dx = dir_vec[self.agent_dir]
        front_y, front_x = self.agent_pos[0] + dy, self.agent_pos[1] + dx

        if not (0 <= front_y < self.size and 0 <= front_x < self.size):
            return

        if self.grid[front_y, front_x, 0] == Objects.KEY:
            self.carrying = self.grid[front_y, front_x, 1]
            self.grid[front_y, front_x, 0] = Objects.EMPTY
            if (front_y, front_x) in self.key_positions:
                del self.key_positions[(front_y, front_x)]
            if self.verbose:
                print(f"Picked up {COLOR_NAMES.get(self.carrying, 'unknown')} key!")

    def _toggle(self):
        """Toggle door in front"""
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy, dx = dir_vec[self.agent_dir]
        front_y, front_x = self.agent_pos[0] + dy, self.agent_pos[1] + dx
        if not (0 <= front_y < self.size and 0 <= front_x < self.size):
            return
        if self.grid[front_y, front_x, 0] == Objects.DOOR:
            state = self.grid[front_y, front_x, 2]
            color = self.grid[front_y, front_x, 1]
            if state == 2:  # Locked
                if self.carrying == color:
                    self.grid[front_y, front_x, 2] = 0
                    # UPDATE door_positions too!
                    if (front_y, front_x) in self.door_positions:
                        self.door_positions[(front_y, front_x)] = (color, 0)
                    self.carrying = None
                    if self.verbose:
                        print(f"Unlocked {COLOR_NAMES.get(color, 'unknown')} door!")
            elif state == 1:  # Closed
                self.grid[front_y, front_x, 2] = 0
                # UPDATE door_positions too!
                if (front_y, front_x) in self.door_positions:
                    self.door_positions[(front_y, front_x)] = (color, 0)
                if self.verbose:
                    print("Opened door!")

    def render(self):
        """Render the environment"""
        if self.render_mode == "console":
            self._render_console()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_pygame()
        return None

    def _render_console(self):
        """ASCII rendering"""
        symbols = {
            Objects.EMPTY: ".",
            Objects.WALL: "#",
            Objects.DOOR: "D",
            Objects.KEY: "K",
            Objects.GOAL: "G",
        }
        dir_symbols = [">", "v", "<", "^"]

        print(f"\nStep: {self.step_count}/{self.max_steps}")
        print(f"Mission: {self.mission}")
        print(f"Carrying: {COLOR_NAMES.get(self.carrying, 'nothing')}")

        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if [y, x] == self.agent_pos:
                    row += dir_symbols[self.agent_dir]
                else:
                    row += symbols.get(self.grid[y, x, 0], "?")
            print(row)

    def _render_pygame(self):
        """Pygame rendering"""
        if self.window is None:
            pygame.init()
            window_size = self.size * self.cell_size
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Locked Room Environment")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Draw grid
        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(
                    x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
                )
                obj = self.grid[y, x, 0]
                color_idx = self.grid[y, x, 1]
                state = self.grid[y, x, 2]

                if obj == Objects.WALL:
                    pygame.draw.rect(self.window, (64, 64, 64), rect)
                elif obj == Objects.DOOR:
                    color = COLOR_MAP.get(color_idx, (128, 128, 128))
                    pygame.draw.rect(self.window, color, rect)
                    if state == 2:
                        pygame.draw.circle(self.window, (0, 0, 0), rect.center, 5)
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 2)
                elif obj == Objects.KEY:
                    pygame.draw.rect(self.window, (240, 240, 240), rect)
                    color = COLOR_MAP.get(color_idx, (128, 128, 128))
                    pygame.draw.circle(self.window, color, rect.center, 8)
                elif obj == Objects.GOAL:
                    pygame.draw.rect(self.window, (0, 255, 0), rect)
                else:
                    pygame.draw.rect(self.window, (240, 240, 240), rect)

                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.cell_size,
            self.agent_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.circle(self.window, (255, 0, 0), agent_rect.center, self.cell_size // 3)

        # Direction indicator
        dir_offsets = [(8, 0), (0, 8), (-8, 0), (0, -8)]
        dx, dy = dir_offsets[self.agent_dir]
        end_pos = (agent_rect.centerx + dx, agent_rect.centery + dy)
        pygame.draw.line(self.window, (255, 255, 0), agent_rect.center, end_pos, 3)

        # Carrying indicator
        if self.carrying is not None:
            carry_color = COLOR_MAP.get(self.carrying, (128, 128, 128))
            pygame.draw.circle(
                self.window, carry_color, (agent_rect.centerx, agent_rect.top - 5), 5
            )

        pygame.display.flip()

        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), (1, 0, 2))

    def close(self):
        """Clean up"""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
