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

REWARD_GOAL = 500.0  # Terminal success reward (reduced for better proportion to milestones)
REWARD_CLOSER = 5.0  # Reward for achieving new best distance to current target (meaningful progress)
REWARD_CLOSER_TO_EXIT = 2.0  # Reward for moving toward room exit when stuck in empty room
PENALTY_FURTHER = 0.0  # Not used - progress tracked via best distance
PENALTY_STEP = -0.5  # Time penalty to encourage efficiency
PENALTY_NO_MOVE = -5.0  # Heavy penalty for not moving (hitting walls or staying still)
REWARD_TURN = 0.0  # Not used in omnidirectional movement
REWARD_KEY_PICKUP = 100.0  # Major milestone reward (reduced to balance with goal)
REWARD_DOOR_OPEN = 100.0  # Major milestone reward (reduced to balance with goal)

# New exploration rewards
REWARD_EXPLORATION = 2  # Small bonus for seeing new cells (prevents over-valuing wandering)
PENALTY_EMPTY_ROOM = -0.1  #Penalty per step in fully explored empty room (encourages leaving)
REWARD_USEFUL_ROOM = 0.0  # Small reward per step in room with objective (encourages staying on task)

# Anti-oscillation penalties
PENALTY_REVISIT = -1.5  # Meaningful penalty for repeat visits (scaled with visit count)
PENALTY_DIRECTION_CHANGE = 0.0  # Disabled - no penalty for changing direction
PENALTY_REVERSAL = -5.0  # Strong penalty for 180° direction reversal (prevent back-and-forth)


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


class LegacySimpleObs(gym.ObservationWrapper):
    """
    Legacy simple observation wrapper (10 features) for backward compatibility with old models.
    
    Observation space (10 features):
    - agent_y, agent_x (2)
    - goal_y, goal_x (2)
    - has_key (1 if carrying any key, 0 otherwise) (1)
    - key_y, key_x (position of first key, or -1 if none/picked up) (2)
    - door_y, door_x (position of first door, or -1 if none) (2)
    - door_locked (1 if locked, 0 if open, -1 if no door) (1)
    
    Total: 10 features
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-1, 20, (10,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_y, agent_x = base_env.agent_pos
        goal_y, goal_x = base_env.goal_pos

        # Do we have any key?
        has_key = 1.0 if base_env.carrying is not None else 0.0

        # First key position (if any)
        key_y, key_x = -1.0, -1.0
        if hasattr(base_env, "key_positions") and base_env.key_positions:
            first_key = list(base_env.key_positions.keys())[0]
            key_y, key_x = float(first_key[0]), float(first_key[1])

        # First door position and state
        door_y, door_x = -1.0, -1.0
        door_locked = -1.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            first_door = list(base_env.door_positions.keys())[0]
            door_y, door_x = float(first_door[0]), float(first_door[1])
            color, state = base_env.door_positions[first_door]
            # state: 0=open, 1=closed, 2=locked
            door_locked = 1.0 if state == 2 else 0.0

        return np.array(
            [
                float(agent_y),
                float(agent_x),
                float(goal_y),
                float(goal_x),
                has_key,
                key_y,
                key_x,
                door_y,
                door_x,
                door_locked,
            ],
            dtype=np.float32,
        )


class SimpleObs(gym.ObservationWrapper):
    """
    Observation wrapper that provides positions of ALL keys and doors.
    
    Observation space (34 features):
    - agent_y, agent_x (2)
    - goal_y, goal_x (2)
    - carrying (color of key being carried, or -1) (1)
    - key_1_y, key_1_x, key_1_color, key_1_visible (4)
    - key_2_y, key_2_x, key_2_color, key_2_visible (4)
    - key_3_y, key_3_x, key_3_color, key_3_visible (4)
    - key_4_y, key_4_x, key_4_color, key_4_visible (4)
    - door_1_y, door_1_x, door_1_color, door_1_state (4)
    - door_2_y, door_2_x, door_2_color, door_2_state (4)
    - door_3_y, door_3_x, door_3_color, door_3_state (4)
    - door_4_y, door_4_x, door_4_color, door_4_state (4)
    
    Total: 2 + 2 + 1 + 4*4 + 4*4 = 37 features (but we use 34 by removing redundant info)
    
    NOTE: This wrapper discards the 7x7 partial view and explored_map from the base environment.
    For agents that need spatial awareness, use EnhancedObs instead.
    """

    def __init__(self, env):
        super().__init__(env)
        # Agent (2) + Goal (2) + Carrying (1) + Keys (4*4=16) + Doors (4*3=12) = 33
        # Adding 1 padding for total of 34
        self.observation_space = spaces.Box(-1, 20, (34,), np.float32)
        self.max_keys = 4
        self.max_doors = 4

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_y, agent_x = base_env.agent_pos
        goal_y, goal_x = base_env.goal_pos

        # What key are we carrying? (-1 if none, otherwise color)
        carrying = float(base_env.carrying) if base_env.carrying is not None else -1.0

        obs_list = [
            float(agent_y),
            float(agent_x),
            float(goal_y),
            float(goal_x),
            carrying,
        ]

        # Add key information (up to 4 keys)
        keys = []
        if hasattr(base_env, "key_positions") and base_env.key_positions:
            for key_pos, key_color in base_env.key_positions.items():
                keys.append({
                    'y': float(key_pos[0]),
                    'x': float(key_pos[1]),
                    'color': float(key_color),
                    'visible': 1.0  # Key is still on grid (not picked up)
                })
        
        # Pad to exactly 4 keys
        while len(keys) < self.max_keys:
            keys.append({'y': -1.0, 'x': -1.0, 'color': -1.0, 'visible': 0.0})
        
        # Add first 4 keys to observation
        for i in range(self.max_keys):
            obs_list.extend([keys[i]['y'], keys[i]['x'], keys[i]['color'], keys[i]['visible']])

        # Add door information (up to 4 doors)
        doors = []
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos, (door_color, door_state) in base_env.door_positions.items():
                doors.append({
                    'y': float(door_pos[0]),
                    'x': float(door_pos[1]),
                    'color': float(door_color),
                    'state': float(door_state)  # 0=open, 1=closed, 2=locked
                })
        
        # Pad to exactly 4 doors
        while len(doors) < self.max_doors:
            doors.append({'y': -1.0, 'x': -1.0, 'color': -1.0, 'state': -1.0})
        
        # Add first 4 doors to observation (using 3 features per door: y, x, state)
        # Color is implicit from position, so we skip it to save space
        for i in range(self.max_doors):
            obs_list.extend([doors[i]['y'], doors[i]['x'], doors[i]['state']])

        # Pad to 34 total features
        while len(obs_list) < 34:
            obs_list.append(0.0)

        return np.array(obs_list[:34], dtype=np.float32)


class EnhancedObs(gym.ObservationWrapper):
    """
    Enhanced observation that provides complete information:
    1. Agent position (2 features) - where am I?
    2. Carrying key color (1 feature) - what am I holding?
    3. Flattened 7x7 partial view (147 features) - what's around me?
    4. Full explored map (1083 features) - what have I seen?
       - Encodes: walls, doors (with colors and states), keys (with colors), goal
    
    Total: 2 + 1 + 147 + 1083 = 1233 features
    
    This provides complete spatial awareness with memory of explored areas.
    """

    def __init__(self, env):
        super().__init__(env)
        # Agent pos (2) + carrying (1) + 7x7 view (147) + explored map (19x19x3 = 1083) = 1233
        self.observation_space = spaces.Box(-1, 255, (1233,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        # 1. Agent position (2 features)
        agent_y, agent_x = base_env.agent_pos
        agent_pos = np.array([float(agent_y), float(agent_x)], dtype=np.float32)

        # 2. Carrying key color (1 feature: -1 if none, else color index)
        carrying = float(base_env.carrying) if base_env.carrying is not None else -1.0
        carrying_array = np.array([carrying], dtype=np.float32)

        # 3. Flatten 7x7 partial view (147 features)
        # This shows immediate surroundings: walls, doors, keys, goal
        if isinstance(obs, dict) and 'image' in obs:
            partial_view = obs['image'].flatten().astype(np.float32)
        else:
            partial_view = np.zeros(7 * 7 * 3, dtype=np.float32)

        # 4. Full explored map (1083 features = 19x19x3)
        # This is the memory of what the agent has seen:
        # - Channel 0: object type (0=empty, 1=wall, 2=door, 3=key, 4=goal)
        # - Channel 1: color (for keys and doors)
        # - Channel 2: state (for doors: 0=open, 1=closed, 2=locked)
        if isinstance(obs, dict) and 'explored_map' in obs:
            explored_map = obs['explored_map'].flatten().astype(np.float32)
        else:
            explored_map = np.zeros(19 * 19 * 3, dtype=np.float32)

        # Combine all features
        full_obs = np.concatenate([
            agent_pos,        # 2
            carrying_array,   # 1
            partial_view,     # 147
            explored_map      # 1083
        ])
        
        return full_obs


# =============================================================================
# REWARD WRAPPER
# =============================================================================


class SimpleRewardWrapper(gym.RewardWrapper):
    """
    Three-phase reward: key -> door -> goal
    All bonuses are one-time only to prevent exploitation.
    Includes exploration bonus and room utility penalties.
    
    Anti-exploitation mechanisms:
    1. Best-distance tracking: Only reward when achieving new minimum distance
    2. Position history: Penalize revisiting recent positions
    3. Movement momentum: Penalize frequent direction changes
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        self.best_distance = None  # Track best (minimum) distance achieved
        # One-time bonus flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        # Repeat action tracking
        self.last_action = None
        self.repeat_count = 0
        # Exploration tracking
        self.explored_cells = set()  # Track all cells agent has seen
        self.fully_explored_rooms = set()  # Track rooms that are fully explored
        self.room_contents = {}  # Cache what's in each room {room_idx: has_useful_item}
        # Anti-oscillation tracking
        self.position_history = []  # Track recent positions to detect oscillation
        self.max_history_length = 50  # Keep last 10 positions
        self.last_direction = None  # Track movement direction for momentum

    def _get_base_env(self):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        return base_env

    def _get_current_target(self):
        """Return current target: accessible key -> door we can open -> goal. Only considers discovered objects."""
        base_env = self._get_base_env()
        
        # First, check if agent is in a fully explored empty room - if so, target the exit
        current_pos = tuple(base_env.agent_pos)
        current_room = self._get_room_index(current_pos)
        
        if current_room != -1:  # Not in corridor
            if self._is_room_fully_explored(current_room):
                if not self._room_has_useful_item(current_room):
                    # In a fully explored empty room - target the door/exit
                    exit_door = self._get_room_door(current_room)
                    if exit_door and self._is_position_explored(exit_door):
                        return exit_door

        # Phase 1: Not carrying a key -> find an ACCESSIBLE key (if any exist and are discovered)
        if base_env.carrying is None:
            if hasattr(base_env, "key_positions") and base_env.key_positions:
                accessible_key = self._find_accessible_key(base_env)
                if accessible_key and self._is_position_explored(accessible_key):
                    return accessible_key
            # No discovered keys -> check if goal is discovered, otherwise return agent position
            if self._is_position_explored(base_env.goal_pos):
                return base_env.goal_pos
            else:
                # Goal not discovered yet - no valid target, return current position
                return tuple(base_env.agent_pos)

        # Phase 2: Carrying a key -> find the matching locked door (if discovered)
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos, (door_color, door_state) in base_env.door_positions.items():
                if door_state == 2 and door_color == base_env.carrying:
                    if self._is_position_explored(door_pos):
                        return door_pos

        # Phase 3: No key needed or door already open -> go to goal (if discovered)
        if self._is_position_explored(base_env.goal_pos):
            return base_env.goal_pos

        # No valid discovered target
        return tuple(base_env.agent_pos)

    def _is_position_explored(self, pos):
        """Check if a position has been explored (seen by the agent)."""
        if pos is None:
            return False
        return pos in self.explored_cells

    def _find_accessible_key(self, base_env):
        """Find a key that is not behind a locked door AND has been discovered."""
        # Build set of rooms blocked by locked doors
        blocked_rooms = set()

        # Door position to room index mapping
        door_to_room = {(3, 7): 0, (9, 7): 1, (15, 7): 2, (3, 11): 3, (9, 11): 4, (15, 11): 5}

        if hasattr(base_env, "door_positions"):
            for door_pos, (color, state) in base_env.door_positions.items():
                if state == 2:  # Locked
                    if door_pos in door_to_room:
                        blocked_rooms.add(door_to_room[door_pos])

        # Find a key not in a blocked room AND that has been discovered
        for key_pos, key_color in base_env.key_positions.items():
            # Only consider keys that have been explored
            if not self._is_position_explored(key_pos):
                continue
                
            key_room = self._get_room_index(key_pos)
            if key_room not in blocked_rooms:
                return key_pos

        # No accessible discovered key found
        return None

    def _get_room_door(self, room_idx):
        """Get the door position for a given room index."""
        # Map room indices to door positions
        room_to_door = {
            0: (3, 7),   # left_top
            1: (9, 7),   # left_middle
            2: (15, 7),  # left_bottom
            3: (3, 11),  # right_top
            4: (9, 11),  # right_middle
            5: (15, 11)  # right_bottom
        }
        return room_to_door.get(room_idx)

    def _get_room_index(self, pos):
        """Get room index for a position (matches env logic)."""
        y, x = pos

        # Corridor
        if 7 <= x <= 11:
            return -1

        # Left side rooms (y boundaries: 1-5, 7-11, 13-17 to exclude wall rows 6, 12)
        if x < 7:
            if y < 6:  # Changed from y <= 5 to be more explicit
                return 0
            if 7 <= y < 12:  # Exclude wall row 6, include 7-11
                return 1
            if y >= 13:  # Exclude wall row 12, include 13+
                return 2

        # Right side rooms
        if x > 11:
            if y < 6:
                return 3
            if 7 <= y < 12:
                return 4
            if y >= 13:
                return 5

        return -1

    def _is_room_fully_explored(self, room_idx):
        """Check if a room is fully explored by checking all its cells."""
        if room_idx == -1:  # Corridor
            return False

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
            return False

        y1, y2, x1, x2 = bounds[room_idx]

        # Check if all cells in room are explored
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if (y, x) not in self.explored_cells:
                    return False

        return True

    def _room_has_useful_item(self, room_idx):
        """Check if room contains key, door, or goal that is still relevant."""
        if room_idx == -1:
            return False

        base_env = self._get_base_env()

        # Define room bounds
        bounds = {
            0: (1, 5, 1, 6),
            1: (7, 11, 1, 6),
            2: (13, 17, 1, 6),
            3: (1, 5, 12, 17),
            4: (7, 11, 12, 17),
            5: (13, 17, 12, 17)
        }

        if room_idx not in bounds:
            return False

        y1, y2, x1, x2 = bounds[room_idx]

        # Check for goal
        if base_env.goal_pos:
            gy, gx = base_env.goal_pos
            if y1 <= gy <= y2 and x1 <= gx <= x2:
                return True

        # Check for keys that are ACTUALLY STILL ON THE GRID (not picked up)
        # A key is still useful only if it's in key_positions AND on the grid
        if hasattr(base_env, 'key_positions') and base_env.key_positions:
            for (ky, kx), key_color in base_env.key_positions.items():
                if y1 <= ky <= y2 and x1 <= kx <= x2:
                    # Verify the key is actually still on the grid
                    # Grid encoding: [object_type, color, state]
                    # Object type 3 = KEY
                    if base_env.grid[ky, kx, 0] == 3:
                        return True

        # Check for locked doors that we need to open
        # Only count doors as useful if they're still locked (state 2)
        if hasattr(base_env, 'door_positions') and base_env.door_positions:
            for (dy, dx), (door_color, door_state) in base_env.door_positions.items():
                # Only count if door is in/near this room AND still locked
                if y1 <= dy <= y2 and x1 <= dx <= x2:
                    if door_state == 2:  # Still locked
                        return True

        return False

    def _update_explored_cells(self, obs):
        """Update set of explored cells from observation."""
        if 'explored_map' not in obs:
            return 0

        explored_map = obs['explored_map']
        base_env = self._get_base_env()
        
        new_cells = 0
        for y in range(base_env.size):
            for x in range(base_env.size):
                # Cell is explored if it's not all zeros in explored_map
                if np.any(explored_map[y, x] > 0):
                    if (y, x) not in self.explored_cells:
                        self.explored_cells.add((y, x))
                        new_cells += 1
        
        return new_cells

    def _distance_to_target(self):
        """
        Calculate BFS distance to current target.
        Returns 0 if no valid target exists (nothing discovered yet).
        """
        base_env = self._get_base_env()
        pos = tuple(base_env.agent_pos)  # Convert to tuple for hashing
        target = self._get_current_target()
        
        # If target is current position (no valid target), return 0
        if target == pos:
            return 0

        from collections import deque

        queue = deque([(pos, 0)])  # (position, distance)
        visited = {pos}
        
        # For doors we need to TOGGLE, we need to reach adjacent cells (to toggle them)
        # For keys, we need to be ON TOP of them (reach the exact position)
        # For exit doors (just passing through), we need to reach the door itself
        need_adjacent = False
        
        # Check if targeting a door we need to unlock (not an exit door)
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos, (door_color, door_state) in base_env.door_positions.items():
                # Only need adjacent if we're targeting a locked door we can unlock
                if door_state == 2 and door_color == base_env.carrying and target == door_pos:
                    if self._is_position_explored(door_pos):
                        need_adjacent = True
                    break

        while queue:
            current, dist = queue.popleft()
            
            # Check if we reached the goal
            if current == target:
                return dist
            
            # For keys/doors, check if we're adjacent
            if need_adjacent and abs(current[0] - target[0]) + abs(current[1] - target[1]) == 1:
                return dist
            
            # Explore neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dy, current[1] + dx)
                
                if next_pos in visited:
                    continue
                
                # Check bounds
                if not (0 <= next_pos[0] < base_env.size and 0 <= next_pos[1] < base_env.size):
                    continue
                
                # Check if cell is walkable
                if not self._is_walkable(next_pos):
                    continue
                
                visited.add(next_pos)
                queue.append((next_pos, dist + 1))
        
        # No path found
        return float('inf')

    def _facing_target(self):
        """Check if moving toward current target (simplified for omnidirectional)."""
        # Since we have omnidirectional movement, this concept doesn't apply
        # We'll just return True to not break existing logic
        return True

    def _has_open_path(self, start, target):
        """
        Check if there's an open path from start to target using BFS.
        A path is blocked if there are walls or locked doors we can't open.
        """
        base_env = self._get_base_env()
        
        # BFS to find if target is reachable
        from collections import deque
        
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            
            # Check if we reached the target (or adjacent to it for keys/doors)
            if current == target:
                return True
            
            # For keys and doors, being adjacent is enough
            if abs(current[0] - target[0]) + abs(current[1] - target[1]) == 1:
                return True
            
            # Explore neighbors (up, down, left, right)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dy, current[1] + dx)
                
                if next_pos in visited:
                    continue
                
                # Check bounds
                if not (0 <= next_pos[0] < base_env.size and 0 <= next_pos[1] < base_env.size):
                    continue
                
                # Check if cell is walkable
                if not self._is_walkable(next_pos):
                    continue
                
                visited.add(next_pos)
                queue.append(next_pos)
        
        return False

    def _is_walkable(self, pos):
        """
        Check if a position is walkable.
        A cell is walkable if it's not a wall and not a locked door we can't open.
        """
        base_env = self._get_base_env()
        y, x = pos
        
        # Get cell type from grid
        # Grid encoding: [object_type, color, state]
        # Objects: EMPTY=0, WALL=1, DOOR=2, KEY=3, GOAL=4
        cell_type = base_env.grid[y, x, 0]
        
        # Wall check (Objects.WALL = 1)
        if cell_type == 1:
            return False
        
        # Check if it's a door (Objects.DOOR = 2)
        if cell_type == 2:
            # Check door state from grid
            door_state = base_env.grid[y, x, 2]
            door_color = base_env.grid[y, x, 1]
            
            # If door is open (state 0), it's walkable
            if door_state == 0:
                return True
            
            # If door is locked (state 2), check if we have the matching key
            if door_state == 2:
                if base_env.carrying == door_color:
                    return True  # We can open it
                else:
                    return False  # Blocked by locked door
            
            # Door is closed but not locked (state 1) - can walk through
            return True
        
        # Otherwise, it's walkable (empty space, goal, key, etc.)
        return True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._distance_to_target()
        self.best_distance = self.previous_distance  # Initialize best distance
        # Reset all flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        self.last_action = None
        self.repeat_count = 0
        # Reset exploration tracking
        self.explored_cells = set()
        self.fully_explored_rooms = set()
        self.room_contents = {}
        # Reset anti-oscillation tracking
        self.position_history = []
        self.last_direction = None
        # Initialize with current view
        self._update_explored_cells(obs)
        return obs, info

    def step(self, action):
        base_env = self._get_base_env()
        old_pos = tuple(base_env.agent_pos)
        had_key_before = base_env.carrying is not None
        key_color_before = base_env.carrying

        # Track ALL door states before action
        door_states_before = {}
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos in base_env.door_positions:
                door_states_before[door_pos] = base_env.grid[door_pos[0], door_pos[1], 2]

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(base_env.agent_pos)
        moved = old_pos != new_pos
        has_key_now = base_env.carrying is not None

        # Initialize reward breakdown tracking
        reward_breakdown = {
            'step_penalty': 0.0,
            'exploration': 0.0,
            'key_pickup': 0.0,
            'door_open': 0.0,
            'closer_to_target': 0.0,
            'closer_to_exit': 0.0,
            'useful_room': 0.0,
            'empty_room_penalty': 0.0,
            'repeat_action_penalty': 0.0,
            'revisit_penalty': 0.0,
            'direction_change_penalty': 0.0,
            'no_move_penalty': 0.0,
            'goal_reached': 0.0,
        }

        # Goal reached
        if base_reward > 0:
            reward_breakdown['goal_reached'] = REWARD_GOAL
            info['reward_breakdown'] = reward_breakdown
            return obs, REWARD_GOAL, terminated, truncated, info

        reward = PENALTY_STEP
        reward_breakdown['step_penalty'] = PENALTY_STEP

        # Heavy penalty for not moving
        if not moved:
            reward += PENALTY_NO_MOVE
            reward_breakdown['no_move_penalty'] = PENALTY_NO_MOVE

        # Exploration bonus - reward for seeing new cells
        new_cells_count = self._update_explored_cells(obs)
        if new_cells_count > 0:
            exploration_reward = REWARD_EXPLORATION * new_cells_count
            reward += exploration_reward
            reward_breakdown['exploration'] = exploration_reward

        # Key pickup bonus
        if has_key_now and not had_key_before:
            reward += REWARD_KEY_PICKUP
            reward_breakdown['key_pickup'] = REWARD_KEY_PICKUP
            self.previous_distance = self._distance_to_target()
            self.best_distance = self.previous_distance  # Reset best distance for new phase
            # Clear position history when picking up key (new phase)
            self.position_history = []
            info['reward_breakdown'] = reward_breakdown
            return obs, reward, terminated, truncated, info

        # Door open bonus - check ALL doors
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos in base_env.door_positions:
                state_before = door_states_before.get(door_pos, 0)
                state_now = base_env.grid[door_pos[0], door_pos[1], 2]
                if state_before == 2 and state_now == 0:  # Was locked, now open
                    reward += REWARD_DOOR_OPEN
                    reward_breakdown['door_open'] = REWARD_DOOR_OPEN
                    self.previous_distance = self._distance_to_target()
                    self.best_distance = self.previous_distance  # Reset best distance for new phase
                    # Clear position history when opening door (new phase)
                    self.position_history = []
                    info['reward_breakdown'] = reward_breakdown
                    return obs, reward, terminated, truncated, info

        # Track position history for oscillation detection
        if moved:
            self.position_history.append(new_pos)
            if len(self.position_history) > self.max_history_length:
                self.position_history.pop(0)
            
            # Penalty for revisiting recent positions (oscillation detection)
            position_count = self.position_history.count(new_pos)
            # print(f"[DEBUG] Position history: {self.position_history}")
            # print(f"[DEBUG] Current position: {new_pos}, count: {position_count}")
            if position_count > 1:
                # Exponential penalty for repeated visits
                revisit_penalty = PENALTY_REVISIT * (position_count - 1)
                reward += revisit_penalty
                reward_breakdown['revisit_penalty'] = revisit_penalty
                #print(f"[DEBUG] Applied revisit penalty: {revisit_penalty}")
            
            # Calculate movement direction for momentum tracking
            current_direction = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
            
            # Penalty for changing direction (reduces oscillation)
            if self.last_direction is not None and self.last_direction != current_direction:
                # Check if it's a direct reversal (180° turn - moving in exact opposite direction)
                if self.last_direction == (-current_direction[0], -current_direction[1]):
                    direction_penalty = PENALTY_REVERSAL  # Stronger penalty for reversing
                else:
                    direction_penalty = PENALTY_DIRECTION_CHANGE  # Mild penalty for direction change
                reward += direction_penalty
                reward_breakdown['direction_change_penalty'] = direction_penalty
            
            self.last_direction = current_direction

        # Distance reward - ONLY reward when achieving NEW BEST distance
        current_distance = self._distance_to_target()
        
        # Only reward if:
        # 1. Path exists (distance is not infinity)
        # 2. We achieved a NEW BEST (better than any previous distance)
        if current_distance != float('inf') and current_distance < self.best_distance:
            # Check if we're targeting a room exit (lower reward to prevent abuse)
            target = self._get_current_target()
            
            # Check if target is a door from a fully explored empty room
            is_targeting_exit = False
            for room_idx in range(6):  # Check all 6 rooms
                room_door = self._get_room_door(room_idx)
                if target == room_door:
                    # Target is a door - check if that room is fully explored and empty
                    if self._is_room_fully_explored(room_idx) and not self._room_has_useful_item(room_idx):
                        is_targeting_exit = True
                        break
            
            if is_targeting_exit:
                reward += REWARD_CLOSER_TO_EXIT
                reward_breakdown['closer_to_exit'] = REWARD_CLOSER_TO_EXIT
            else:
                reward += REWARD_CLOSER
                reward_breakdown['closer_to_target'] = REWARD_CLOSER
            
            # Update best distance
            self.best_distance = current_distance
        
        self.previous_distance = current_distance

        # Room utility reward (only for useful rooms, empty rooms now target the exit via _get_current_target)
        current_room = self._get_room_index(new_pos)
        if current_room != -1:  # Not in corridor
            # Check if room is fully explored
            if self._is_room_fully_explored(current_room):
                if current_room not in self.fully_explored_rooms:
                    self.fully_explored_rooms.add(current_room)

                # Check if room has useful items (don't cache - items can be picked up!)
                has_useful_items = self._room_has_useful_item(current_room)
                
                # Only give reward for useful rooms (empty rooms handled by targeting exit)
                if has_useful_items:
                    reward += REWARD_USEFUL_ROOM  # Reward for being in useful room
                    reward_breakdown['useful_room'] = REWARD_USEFUL_ROOM

        # Penalty for repeating non-moving actions
        if action == self.last_action and not moved:
            self.repeat_count += 1
            if self.repeat_count > 3:
                reward -= 0.2
                reward_breakdown['repeat_action_penalty'] = -0.2
        else:
            self.repeat_count = 0
        self.last_action = action

        # Add reward breakdown to info
        info['reward_breakdown'] = reward_breakdown

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
