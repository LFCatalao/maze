"""
Customizable Locked Room Environment for Reinforcement Learning
Based on MiniGrid's LockedRoom environment with enhanced flexibility
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
    LEFT = 0      # Turn left
    RIGHT = 1     # Turn right
    FORWARD = 2   # Move forward
    PICKUP = 3    # Pick up object
    TOGGLE = 4    # Toggle/activate object (open door)


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


class LockedRoomEnv(gym.Env):
    """
    Flexible Locked Room Environment
    
    Features:
    - Configurable observation space (full map, partial view, or agent view)
    - Progressive task learning support
    - Pygame rendering
    - Console printing option
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "console"], "render_fps": 10}
    
    def __init__(
        self,
        size: int = 19,
        observation_mode: str = "full_map",
        task_type: str = "full",
        render_mode: Optional[str] = "human",
        max_steps: int = 1000,
        agent_view_size: int = 7,
        include_mission_in_obs: bool = True,
        num_doors: int = 6,
        randomize_doors: bool = True,
        include_key: bool = True,
        locked_door: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the environment
        
        Args:
            size: Size of the grid (default 19x19)
            observation_mode: Type of observation
                - "full_map": Agent sees entire grid
                - "partial_view": Agent sees a local window
                - "agent_view": First-person view (7x7 in front)
            task_type: Type of task
                - "reach_door": Just reach any door
                - "get_key": Get the key
                - "unlock_door": Get key and unlock specific door
                - "full": Complete the full task (reach goal)
            render_mode: Rendering mode ("human", "rgb_array", "console", None)
            max_steps: Maximum steps per episode
            agent_view_size: Size of agent's view (for agent_view mode)
            include_mission_in_obs: Include task type encoding in observations
            num_doors: Number of doors to place (1-6)
            randomize_doors: Whether to randomize door positions
            include_key: Whether to include a key in the environment
            locked_door: Whether one door should be locked
            verbose: Whether to print action feedback messages (True for manual play, False for training)
        """
        super().__init__()
        
        self.size = size
        self.observation_mode = observation_mode
        self.task_type = task_type
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.agent_view_size = agent_view_size
        self.include_mission_in_obs = include_mission_in_obs
        self.num_doors = max(0, min(6, num_doors))  # Allow 0 doors for open navigation
        self.randomize_doors = randomize_doors
        self.include_key = include_key
        self.locked_door = locked_door
        self.verbose = verbose
        
        # Action space
        self.action_space = spaces.Discrete(5)
        
        # Observation space depends on mode
        self._setup_observation_space()
        
        # Pygame setup
        self.cell_size = 32
        self.window = None
        self.clock = None
        
        # Environment state
        self.grid = None
        self.agent_pos = None
        self.agent_dir = 0  # 0: right, 1: down, 2: left, 3: up
        self.carrying = None
        self.step_count = 0
        
        # Room and mission setup
        self.door_positions = {}
        self.key_positions = {}
        self.goal_pos = None
        self.locked_door_color = None
        self.key_room_color = None
        self.mission = ""
        
        # Initialize
        self.reset()
    
    def _setup_observation_space(self):
        """Setup observation space based on mode"""
        base_obs = {}
        
        if self.observation_mode == "full_map":
            # Full grid view: (height, width, 3) - object, color, state
            base_obs["image"] = spaces.Box(
                low=0, high=255,
                shape=(self.size, self.size, 3),
                dtype=np.uint8
            )
        elif self.observation_mode == "partial_view":
            # Partial view around agent
            view_size = 2 * self.agent_view_size + 1
            base_obs["image"] = spaces.Box(
                low=0, high=255,
                shape=(view_size, view_size, 3),
                dtype=np.uint8
            )
        else:  # agent_view
            # First-person view (what agent sees in front)
            base_obs["image"] = spaces.Box(
                low=0, high=255,
                shape=(self.agent_view_size, self.agent_view_size, 3),
                dtype=np.uint8
            )
        
        base_obs["direction"] = spaces.Discrete(4)
        base_obs["carrying"] = spaces.Discrete(len(Colors) + 1)  # +1 for nothing
        
        # Add mission encoding if requested
        if self.include_mission_in_obs:
            # Dynamic mission encoding: 0=get_to_exit, 1=get_key, 2=open_door
            base_obs["mission"] = spaces.Discrete(3)
        
        self.observation_space = spaces.Dict(base_obs)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.step_count = 0
        self.carrying = None
        
        # Generate the locked room layout
        self._generate_locked_room()
        
        # Get observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _generate_locked_room(self):
        """Generate the 6-room layout with central corridor and configurable doors/keys"""
        # Create walls
        self.grid[:, :, 0] = Objects.EMPTY
        
        # Outer walls
        self.grid[0, :, 0] = Objects.WALL
        self.grid[-1, :, 0] = Objects.WALL
        self.grid[:, 0, 0] = Objects.WALL
        self.grid[:, -1, 0] = Objects.WALL
        
        # Create vertical walls at x=7 and x=11 to separate corridor from rooms
        # Corridor is from x=8 to x=10
        for y in range(self.size):
            self.grid[y, 7, 0] = Objects.WALL
            self.grid[y, 11, 0] = Objects.WALL
        
        # Add horizontal walls at y=12 and y=6
        # From x=1 to x=6 (left side rooms)
        for x in range(1, 7):
            self.grid[12, x, 0] = Objects.WALL
            self.grid[6, x, 0] = Objects.WALL
        
        # From x=12 to x=17 (right side rooms)
        for x in range(12, 18):
            self.grid[12, x, 0] = Objects.WALL
            self.grid[6, x, 0] = Objects.WALL
        
        # Define possible door positions with specific colors
        # (y, x, color, state) - state: 0=open, 1=closed, 2=locked
        all_door_positions = [
            (15, 7, Colors.YELLOW, 0),   # Yellow door at x=7, y=15
            (9, 7, Colors.GREEN, 0),     # Green door at x=7, y=9
            (3, 7, Colors.BLUE, 0),      # Blue door at x=7, y=3
            (15, 11, Colors.PURPLE, 0),  # Purple door at x=11, y=15
            (9, 11, Colors.GREY, 0),     # Grey door at x=11, y=9
            (3, 11, Colors.RED, 0),      # Red door at x=11, y=3
        ]
        
        # Select doors based on configuration
        if self.randomize_doors:
            # Randomly select num_doors positions
            indices = np.random.choice(len(all_door_positions), self.num_doors, replace=False)
            door_positions_list = [all_door_positions[i] for i in sorted(indices)]
        else:
            # Use first num_doors positions (fixed)
            door_positions_list = all_door_positions[:self.num_doors]
        
        # If no doors, create openings at all door positions (remove walls)
        if self.num_doors == 0:
            # Create openings at all potential door positions
            for y, x, _, _ in all_door_positions:
                self.grid[y, x, 0] = Objects.EMPTY
        
        # Determine which door to lock (if any)
        if self.locked_door and len(door_positions_list) > 0:
            # Lock a random door (or first one if not randomizing)
            lock_idx = np.random.randint(0, len(door_positions_list)) if self.randomize_doors else 0
            y, x, color, _ = door_positions_list[lock_idx]
            door_positions_list[lock_idx] = (y, x, color, 2)  # Set state to locked
            self.locked_door_color = color
        else:
            self.locked_door_color = door_positions_list[0][2] if door_positions_list else Colors.BLUE
        
        # Place doors
        self.door_positions = {}
        for y, x, color, state in door_positions_list:
            self.grid[y, x, 0] = Objects.DOOR
            self.grid[y, x, 1] = color
            self.grid[y, x, 2] = state
            self.door_positions[(y, x)] = (color, state)
        
        # Place key if requested
        if self.include_key:
            # Place key in one of the accessible rooms on the right side (x > 11)
            key_room_choices = [
                (np.random.randint(13, 18), np.random.randint(12, 18)),  # Top right room
                (np.random.randint(7, 12), np.random.randint(12, 18)),   # Middle right room
                (np.random.randint(1, 6), np.random.randint(12, 18)),    # Bottom right room
            ]
            key_y, key_x = key_room_choices[np.random.randint(0, len(key_room_choices))]
            
            self.grid[key_y, key_x, 0] = Objects.KEY
            self.grid[key_y, key_x, 1] = self.locked_door_color
            self.key_positions[(key_y, key_x)] = self.locked_door_color
            self.key_room_color = Colors.GREY  # For mission description
        else:
            self.key_room_color = Colors.GREY
        
        # Place goal in the locked room (behind blue door, left side)
        goal_x = np.random.randint(1, 7)
        goal_y = np.random.randint(1, 6)
        self.grid[goal_y, goal_x, 0] = Objects.GOAL
        self.goal_pos = (goal_y, goal_x)
        
        # Place agent in the corridor (x=8 to x=10, y=2 to y=18)
        agent_x = np.random.randint(8, 11)  # x from 8 to 10
        agent_y = np.random.randint(2, 18)  # y from 2 to 17
        self.agent_pos = [agent_y, agent_x]
        self.agent_dir = 0  # Facing right
        
        # Generate mission string
        self._generate_mission()
    
    def _generate_mission(self):
        """Generate mission string"""
        locked_color = COLOR_NAMES[self.locked_door_color]
        key_room = COLOR_NAMES[self.key_room_color]
        
        if self.task_type == "reach_door":
            self.mission = f"reach any door"
        elif self.task_type == "get_key":
            self.mission = f"get the {locked_color} key"
        elif self.task_type == "unlock_door":
            self.mission = f"get the {locked_color} key and unlock the {locked_color} door"
        else:  # full
            self.mission = f"get the {locked_color} key from the {key_room} room, unlock the {locked_color} door and go to the goal"
    
    def _get_obs(self) -> Dict:
        """Get current observation based on observation mode"""
        if self.observation_mode == "full_map":
            image = self.grid.copy()
        elif self.observation_mode == "partial_view":
            # Extract partial view around agent
            half_view = self.agent_view_size
            y, x = self.agent_pos
            
            # Create padded grid
            padded_grid = np.zeros(
                (self.size + 2 * half_view, self.size + 2 * half_view, 3),
                dtype=np.uint8
            )
            padded_grid[half_view:-half_view, half_view:-half_view] = self.grid
            
            # Extract view
            image = padded_grid[
                y:y + 2 * half_view + 1,
                x:x + 2 * half_view + 1
            ]
        else:  # agent_view
            # First-person view
            image = self._get_agent_view()
        
        carrying_idx = self.carrying if self.carrying is not None else len(Colors)
        
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "carrying": carrying_idx,
        }
        
        # Add mission encoding if requested
        if self.include_mission_in_obs:
            obs["mission"] = self._compute_current_mission()
        
        return obs
    
    def _compute_current_mission(self) -> int:
        """Compute current mission based on environment state
        
        Returns:
            0: get_to_exit (no door blocking or door is open)
            1: get_key (door exists and is locked/closed, agent doesn't have key)
            2: open_door (agent has the key)
        """
        # Check if there's a locked or closed door blocking the path
        has_blocking_door = False
        for (y, x), (color, state) in self.door_positions.items():
            if state != 0:  # Door is closed (1) or locked (2)
                has_blocking_door = True
                break
        
        if not has_blocking_door:
            # No door blocking, mission is to get to exit
            return 0  # get_to_exit
        
        # There's a blocking door
        if self.carrying == self.locked_door_color:
            # Agent has the key, mission is to open the door
            return 2  # open_door
        else:
            # Agent doesn't have the key, mission is to get it
            return 1  # get_key
    
    def _get_agent_view(self) -> np.ndarray:
        """Get first-person view from agent's perspective"""
        view = np.zeros((self.agent_view_size, self.agent_view_size, 3), dtype=np.uint8)
        
        y, x = self.agent_pos
        
        # Direction vectors
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        dy, dx = dir_vec[self.agent_dir]
        
        # Right vector (perpendicular to direction)
        right_vec = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        ry, rx = right_vec[self.agent_dir]
        
        # Fill view
        for i in range(self.agent_view_size):
            for j in range(self.agent_view_size):
                # Calculate world position
                dist = i
                offset = j - self.agent_view_size // 2
                
                wy = y + dist * dy + offset * ry
                wx = x + dist * dx + offset * rx
                
                if 0 <= wy < self.size and 0 <= wx < self.size:
                    view[i, j] = self.grid[wy, wx]
        
        return view
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "step_count": self.step_count,
            "agent_pos": self.agent_pos,
            "carrying": self.carrying,
            "mission": self.mission,
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        
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
        
        # Check task completion
        terminated, reward = self._check_task_completion()
        
        # Check timeout
        if self.step_count >= self.max_steps:
            truncated = True
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _move_forward(self):
        """Move agent forward if possible"""
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy, dx = dir_vec[self.agent_dir]
        
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx
        
        # Check bounds
        if not (0 <= new_y < self.size and 0 <= new_x < self.size):
            return
        
        # Check what's in front
        obj = self.grid[new_y, new_x, 0]
        
        # Can't walk through walls or closed doors
        if obj == Objects.WALL:
            return
        if obj == Objects.DOOR:
            state = self.grid[new_y, new_x, 2]
            if state != 0:  # Door is closed or locked
                return
        
        # Move
        self.agent_pos = [new_y, new_x]
    
    def _pickup(self):
        """Pick up object in front of agent"""
        if self.carrying is not None:
            if self.verbose:
                print(f"Already carrying {COLOR_NAMES.get(self.carrying, 'something')} key!")
            return
        
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy, dx = dir_vec[self.agent_dir]
        
        front_y = self.agent_pos[0] + dy
        front_x = self.agent_pos[1] + dx
        
        if not (0 <= front_y < self.size and 0 <= front_x < self.size):
            return
        
        obj = self.grid[front_y, front_x, 0]
        
        if obj == Objects.KEY:
            # Pick up key
            key_color = self.grid[front_y, front_x, 1]
            self.carrying = key_color
            self.grid[front_y, front_x, 0] = Objects.EMPTY
            if (front_y, front_x) in self.key_positions:
                del self.key_positions[(front_y, front_x)]
            if self.verbose:
                print(f"\n*** Picked up {COLOR_NAMES.get(key_color, 'unknown')} key! ***")
        else:
            if self.verbose:
                if obj == Objects.EMPTY:
                    print("Nothing to pick up here.")
                else:
                    print(f"Can't pick up that object.")
    
    def _toggle(self):
        """Toggle/activate object in front of agent (e.g., open door)"""
        dir_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy, dx = dir_vec[self.agent_dir]
        
        front_y = self.agent_pos[0] + dy
        front_x = self.agent_pos[1] + dx
        
        if not (0 <= front_y < self.size and 0 <= front_x < self.size):
            return
        
        obj = self.grid[front_y, front_x, 0]
        
        if obj == Objects.DOOR:
            color = self.grid[front_y, front_x, 1]
            state = self.grid[front_y, front_x, 2]
            door_color_name = COLOR_NAMES.get(color, 'unknown')
            
            # If locked, need matching key
            if state == 2:
                if self.carrying == color:
                    self.grid[front_y, front_x, 2] = 0  # Unlock and open
                    self.carrying = None  # Use up key
                    if self.verbose:
                        print(f"\n*** Unlocked and opened {door_color_name} door! ***")
                else:
                    if self.verbose:
                        if self.carrying is not None:
                            carrying_name = COLOR_NAMES.get(self.carrying, 'unknown')
                            print(f"This {door_color_name} door is locked! You have {carrying_name} key but need {door_color_name} key.")
                        else:
                            print(f"This {door_color_name} door is locked! You need the {door_color_name} key.")
            # If closed, just open
            elif state == 1:
                self.grid[front_y, front_x, 2] = 0
                if self.verbose:
                    print(f"\n*** Opened {door_color_name} door! ***")
            else:
                if self.verbose:
                    print(f"Door is already open.")
        else:
            if self.verbose:
                if obj == Objects.EMPTY:
                    print("Nothing to toggle here.")
                else:
                    print(f"Can't toggle that object.")
    
    def _check_task_completion(self) -> Tuple[bool, float]:
        """Check if current task is completed"""
        reward = 0
        terminated = False
        
        # Simple binary reward: 1 if agent reaches goal, 0 otherwise
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 1.0
            terminated = True
        
        return terminated, reward
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "console":
            self._render_console()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_pygame()
        return None
    
    def _render_console(self):
        """Render to console as ASCII"""
        print("\n" + "="*50)
        print(f"Step: {self.step_count}/{self.max_steps}")
        print(f"Mission: {self.mission}")
        print(f"Carrying: {COLOR_NAMES.get(self.carrying, 'nothing')}")
        print(f"Position: {self.agent_pos}, Direction: {self.agent_dir}")
        
        # Create ASCII representation
        symbols = {
            Objects.EMPTY: '.',
            Objects.WALL: '#',
            Objects.DOOR: 'D',
            Objects.KEY: 'K',
            Objects.GOAL: 'G',
        }
        
        dir_symbols = ['>', 'v', '<', '^']
        
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if [y, x] == self.agent_pos:
                    row += dir_symbols[self.agent_dir]
                else:
                    obj = self.grid[y, x, 0]
                    row += symbols.get(obj, '?')
            print(row)
        print("="*50)
    
    def _render_pygame(self):
        """Render using pygame"""
        if self.window is None:
            pygame.init()
            window_size = self.size * self.cell_size
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Locked Room Environment")
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((255, 255, 255))
        
        # Draw grid
        for y in range(self.size):
            for x in range(self.size):
                obj = self.grid[y, x, 0]
                color_idx = self.grid[y, x, 1]
                state = self.grid[y, x, 2]
                
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Draw based on object type
                if obj == Objects.WALL:
                    pygame.draw.rect(self.window, (64, 64, 64), rect)
                elif obj == Objects.DOOR:
                    color = COLOR_MAP.get(color_idx, (128, 128, 128))
                    if state == 2:  # Locked
                        pygame.draw.rect(self.window, color, rect)
                        pygame.draw.rect(self.window, (0, 0, 0), rect, 2)
                        # Draw lock symbol
                        center = rect.center
                        pygame.draw.circle(self.window, (0, 0, 0), center, 5)
                    elif state == 1:  # Closed
                        pygame.draw.rect(self.window, color, rect)
                        pygame.draw.rect(self.window, (0, 0, 0), rect, 2)
                    else:  # Open
                        pygame.draw.rect(self.window, (200, 200, 200), rect)
                elif obj == Objects.KEY:
                    color = COLOR_MAP.get(color_idx, (128, 128, 128))
                    pygame.draw.rect(self.window, (240, 240, 240), rect)
                    center = rect.center
                    pygame.draw.circle(self.window, color, center, 8)
                elif obj == Objects.GOAL:
                    pygame.draw.rect(self.window, (0, 255, 0), rect)
                    pygame.draw.rect(self.window, (0, 200, 0), rect, 3)
                else:  # Empty
                    pygame.draw.rect(self.window, (240, 240, 240), rect)
                
                # Draw grid lines
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)
        
        # Draw agent
        agent_y, agent_x = self.agent_pos
        agent_rect = pygame.Rect(
            agent_x * self.cell_size,
            agent_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.circle(self.window, (255, 0, 0), agent_rect.center, self.cell_size // 3)
        
        # Draw direction indicator
        dir_offsets = [(8, 0), (0, 8), (-8, 0), (0, -8)]
        dx, dy = dir_offsets[self.agent_dir]
        end_pos = (agent_rect.centerx + dx, agent_rect.centery + dy)
        pygame.draw.line(self.window, (255, 255, 0), agent_rect.center, end_pos, 3)
        
        # Draw carrying indicator
        if self.carrying is not None:
            carry_color = COLOR_MAP.get(self.carrying, (128, 128, 128))
            carry_pos = (agent_rect.centerx, agent_rect.top - 5)
            pygame.draw.circle(self.window, carry_color, carry_pos, 5)
        
        # Update display
        pygame.display.flip()
        
        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])
        
        # Return RGB array if needed
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
