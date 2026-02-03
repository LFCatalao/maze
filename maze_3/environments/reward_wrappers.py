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

REWARD_GOAL = 100.0
REWARD_CLOSER = 2
REWARD_CLOSER_TO_EXIT = 0.2  # Reduced reward for moving toward room exit (prevent abuse)
PENALTY_FURTHER = 0.0  # Removed - already captured by REWARD_CLOSER
PENALTY_STEP = -0.4
REWARD_TURN = 0.0
REWARD_KEY_PICKUP = 20.0
REWARD_DOOR_OPEN = 20.0

# New exploration rewards
REWARD_EXPLORATION = 0.2  # Bonus for seeing a new cell
PENALTY_EMPTY_ROOM = 0  # Penalty per step in fully explored empty room
REWARD_USEFUL_ROOM = 0  # Reward per step in room with objective


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
    """Simple observation with key support - updated for omnidirectional movement."""

    def __init__(self, env):
        super().__init__(env)
        # Reduced from 12 to 10 since we removed direction and direction-dependent fields
        self.observation_space = spaces.Box(0, 20, (10,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        agent_y, agent_x = base_env.agent_pos
        goal_y, goal_x = base_env.goal_pos

        # Key info
        has_key = 1.0 if base_env.carrying is not None else 0.0

        # Key position
        key_y, key_x = -1.0, -1.0
        if hasattr(base_env, "key_positions") and base_env.key_positions:
            if base_env.carrying is None:
                key_pos = list(base_env.key_positions.keys())[0]
                key_y, key_x = float(key_pos[0]), float(key_pos[1])

        # Door position
        door_y, door_x = -1.0, -1.0
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            door_pos = list(base_env.door_positions.keys())[0]
            door_y, door_x = float(door_pos[0]), float(door_pos[1])

        return np.array(
            [
                agent_y,
                agent_x,
                goal_y,
                goal_x,
                has_key,
                key_y,
                key_x,
                door_y,
                door_x,
                0.0,  # Placeholder for compatibility
            ],
            dtype=np.float32,
        )


# =============================================================================
# REWARD WRAPPER
# =============================================================================


class SimpleRewardWrapper(gym.RewardWrapper):
    """
    Three-phase reward: key -> door -> goal
    All bonuses are one-time only to prevent exploitation.
    Includes exploration bonus and room utility penalties.
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
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

        # Left side rooms
        if x < 7:
            if y <= 5:
                return 0
            if y <= 11:
                return 1
            return 2

        # Right side rooms
        if x > 11:
            if y <= 5:
                return 3
            if y <= 11:
                return 4
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
        
        # For keys and doors, we need to reach adjacent cells
        need_adjacent = False
        if (
            base_env.carrying is None
            and hasattr(base_env, "key_positions")
            and base_env.key_positions
        ):
            need_adjacent = True
        elif hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos, (door_color, door_state) in base_env.door_positions.items():
                if door_state == 2 and door_color == base_env.carrying:
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
        # Reset all flags
        self.gave_pickup_hint = False
        self.gave_door_hint = False
        self.last_action = None
        self.repeat_count = 0
        # Reset exploration tracking
        self.explored_cells = set()
        self.fully_explored_rooms = set()
        self.room_contents = {}
        # Initialize with current view
        self._update_explored_cells(obs)
        return obs, info

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)
        base_env = self._get_base_env()
        had_key_before = base_env.carrying is not None
        key_color_before = base_env.carrying

        # Track ALL door states before action
        door_states_before = {}
        if hasattr(base_env, "door_positions") and base_env.door_positions:
            for door_pos in base_env.door_positions:
                door_states_before[door_pos] = base_env.grid[door_pos[0], door_pos[1], 2]

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        moved = old_pos != new_pos
        has_key_now = base_env.carrying is not None

        # Initialize reward breakdown tracking
        reward_breakdown = {
            'step_penalty': 0.0,
            'exploration': 0.0,
            'key_pickup': 0.0,
            'door_open': 0.0,
            'closer_to_target': 0.0,
            'useful_room': 0.0,
            'empty_room_penalty': 0.0,
            'repeat_action_penalty': 0.0,
            'goal_reached': 0.0,
        }

        # Goal reached
        if base_reward > 0:
            reward_breakdown['goal_reached'] = REWARD_GOAL
            info['reward_breakdown'] = reward_breakdown
            return obs, REWARD_GOAL, terminated, truncated, info

        reward = PENALTY_STEP
        reward_breakdown['step_penalty'] = PENALTY_STEP

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
                    info['reward_breakdown'] = reward_breakdown
                    return obs, reward, terminated, truncated, info

        # Distance reward (only if actual path distance decreased)
        current_distance = self._distance_to_target()
        
        # Only reward if path distance decreased and path is not blocked (distance is not infinity)
        if current_distance != float('inf') and current_distance < self.previous_distance:
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
                reward_breakdown['closer_to_target'] = REWARD_CLOSER_TO_EXIT
            else:
                reward += REWARD_CLOSER
                reward_breakdown['closer_to_target'] = REWARD_CLOSER
        
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
