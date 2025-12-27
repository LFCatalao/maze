"""
Reward Wrappers for RL Comparison Study

Different reward strategies to compare:
- R1: Sparse (original) - only +1 at goal
- R2: Distance-based - reward for getting closer
- R3: Step penalty - time pressure
- R4: Combined - distance shaping + goal bonus
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional


class SparseRewardWrapper(gym.RewardWrapper):
    """
    R1: Sparse Reward

    +1.0 when reaching the goal, 0 otherwise.
    This is the hardest to learn from due to no intermediate feedback.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_type = "R1_sparse"

    def reward(self, reward):
        # The base environment already returns sparse reward
        # Just pass it through
        return reward


class DistanceRewardWrapper(gym.RewardWrapper):
    """
    R2: Distance-based Reward

    Rewards the agent for getting closer to the current objective.
    Provides dense feedback to guide learning.

    Reward = (previous_distance - current_distance) * scale
    """

    def __init__(self, env, scale: float = 0.1, goal_bonus: float = 1.0):
        super().__init__(env)
        self.reward_type = "R2_distance"
        self.scale = scale
        self.goal_bonus = goal_bonus
        self.previous_distance = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._compute_distance()
        return obs, info

    def _compute_distance(self) -> float:
        """Compute Manhattan distance to current objective"""
        agent_pos = self.env.agent_pos

        # Determine current objective based on state
        if self.env.carrying is None and self.env.include_key and self.env.locked_door:
            # Need to get the key first
            if self.env.key_positions:
                key_pos = list(self.env.key_positions.keys())[0]
                return abs(agent_pos[0] - key_pos[0]) + abs(agent_pos[1] - key_pos[1])

        # Otherwise, go to goal
        goal_pos = self.env.goal_pos
        return abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

    def reward(self, reward):
        current_distance = self._compute_distance()

        distance_reward = (self.previous_distance - current_distance) * self.scale
        self.previous_distance = current_distance

        if reward > 0:  # Goal reached
            return self.goal_bonus

        # Base: distance reward minus small step cost
        total_reward = distance_reward - 0.001

        # Extra penalty for no progress
        if distance_reward == 0:
            total_reward -= 0.01

        return total_reward


class StepPenaltyRewardWrapper(gym.RewardWrapper):
    """
    R3: Step Penalty Reward

    Rewards efficiency - fewer steps = higher reward.
    reward = base_reward * (1 - step_fraction)

    At goal: reward = 1.0 * (1 - 0.9 * steps/max_steps)
    """

    def __init__(self, env, penalty_scale: float = 0.9):
        super().__init__(env)
        self.reward_type = "R3_step_penalty"
        self.penalty_scale = penalty_scale

    def reward(self, reward):
        if reward > 0:  # Goal reached
            step_fraction = self.env.step_count / self.env.max_steps
            return 1.0 - self.penalty_scale * step_fraction

        # Small negative reward per step to encourage efficiency
        return -0.001


class CombinedRewardWrapper(gym.RewardWrapper):
    """
    R4: Combined Reward

    Combines distance-based shaping with goal bonus and step penalty.

    - Distance reward for getting closer to objective
    - Bonus for reaching goal
    - Small step penalty for efficiency
    """

    def __init__(
        self,
        env,
        distance_scale: float = 0.1,
        goal_bonus: float = 1.0,
        step_penalty: float = 0.001,
        key_bonus: float = 0.5,
        door_bonus: float = 0.3,
    ):
        super().__init__(env)
        self.reward_type = "R4_combined"
        self.distance_scale = distance_scale
        self.goal_bonus = goal_bonus
        self.step_penalty = step_penalty
        self.key_bonus = key_bonus
        self.door_bonus = door_bonus

        self.previous_distance = None
        self.had_key = False
        self.doors_opened = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._compute_distance_to_goal()
        self.had_key = False
        self.doors_opened = set()
        self.visited = set()
        return obs, info

    def _compute_distance_to_goal(self) -> float:
        """Compute Manhattan distance to goal"""
        agent_pos = self.env.agent_pos
        goal_pos = self.env.goal_pos
        return abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

    def step(self, action):
        # Track state before step
        had_key_before = self.env.carrying is not None

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate custom reward
        custom_reward = 0.0

        # 1. Distance-based reward
        current_distance = self._compute_distance_to_goal()
        distance_reward = (self.previous_distance - current_distance) * self.distance_scale
        custom_reward += distance_reward
        self.previous_distance = current_distance

        # 2. Key pickup bonus
        has_key_now = self.env.carrying is not None
        if has_key_now and not self.had_key:
            custom_reward += self.key_bonus
            self.had_key = True

        # 3. Goal reached bonus
        if reward > 0:  # Original reward indicates goal
            custom_reward += self.goal_bonus

        # 5. Exploration bonus - reward visiting new positions
        if not hasattr(self, "visited"):
            self.visited = set()

        agent_tuple = tuple(self.env.agent_pos)
        if agent_tuple not in self.visited:
            custom_reward += 0.05  # Bonus for new positions
            self.visited.add(agent_tuple)

        return obs, custom_reward, terminated, truncated, info

    def reward(self, reward):
        # Not used - we override step() instead
        return reward


class EscalatingRewardWrapper(gym.RewardWrapper):
    """
    R5: Escalating Reward with Waypoint Navigation

    The agent follows waypoints:
    1. First, navigate to the door of the goal room
    2. Then, navigate to the goal itself

    This prevents the agent from getting stuck trying to move
    directly toward a goal it can't reach.
    """

    def __init__(self, env, distance_scale: float = 0.1, goal_bonus: float = 1.0):
        super().__init__(env)
        self.reward_type = "R5_escalating"
        self.distance_scale = distance_scale
        self.goal_bonus = goal_bonus

        # State tracking
        self.previous_distance = None
        self.stuck_counter = 0
        self.corridor_counter = 0
        self.visited = set()
        self.steps_taken = 0
        self.previous_room = None
        self.reached_door = False
        self.entered_correct_room = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.stuck_counter = 0
        self.corridor_counter = 0
        self.visited = set()
        self.visited.add(tuple(self.env.agent_pos))
        self.steps_taken = 0
        self.previous_room = self._get_room(self.env.agent_pos)
        self.reached_door = False
        self.entered_correct_room = False

        # Set initial target
        self.door_position = self._get_goal_room_door()
        self.previous_distance = self._compute_distance_to_target()

        return obs, info

    def _get_room(self, pos) -> str:
        """Determine which room/area the agent is in."""
        y, x = pos[0], pos[1]

        if 7 <= x <= 11:
            return "corridor"

        if x < 7:
            if y < 6:
                return "left_top"
            elif y < 12:
                return "left_middle"
            else:
                return "left_bottom"

        if x > 11:
            if y < 6:
                return "right_top"
            elif y < 12:
                return "right_middle"
            else:
                return "right_bottom"

        return "unknown"

    def _get_goal_room(self) -> str:
        """Get the room where the goal is located."""
        return self._get_room(list(self.env.goal_pos))

    def _get_goal_room_door(self) -> tuple:
        """
        Get the door position that leads to the goal room.

        Door positions in the grid:
        - (3, 7): connects corridor to left_top
        - (9, 7): connects corridor to left_middle
        - (15, 7): connects corridor to left_bottom
        - (3, 11): connects corridor to right_top
        - (9, 11): connects corridor to right_middle
        - (15, 11): connects corridor to right_bottom
        """
        goal_room = self._get_goal_room()

        door_map = {
            "left_top": (3, 7),
            "left_middle": (9, 7),
            "left_bottom": (15, 7),
            "right_top": (3, 11),
            "right_middle": (9, 11),
            "right_bottom": (15, 11),
        }

        return door_map.get(goal_room, (9, 9))  # Default to center if unknown

    # def _compute_distance_to_target(self) -> float:
    #     """Compute distance to current target (door or goal)."""
    #     agent_pos = self.env.agent_pos

    #     # If we haven't reached the door yet, target is the door
    #     if not self.entered_correct_room:
    #         target = self.door_position
    #     else:
    #         # Once in the room, target is the goal
    #         target = self.env.goal_pos

    #     return abs(agent_pos[0] - target[0]) + abs(agent_pos[1] - target[1])

    def _compute_distance_to_target(self) -> tuple:
        """
        Compute distance to current target as (dy, dx) tuple.
        Returns separate Y and X distances for independent tracking.
        """
        agent_pos = self.env.agent_pos

        if not self.entered_correct_room:
            target = self.door_position
        else:
            target = self.env.goal_pos

        dy = abs(agent_pos[0] - target[0])
        dx = abs(agent_pos[1] - target[1])

        return (dy, dx)

    def _is_at_door(self) -> bool:
        """Check if agent is at or adjacent to the goal room door."""
        agent_pos = tuple(self.env.agent_pos)
        door = self.door_position

        # At the door
        if agent_pos == door:
            return True

        # Adjacent to door (within 1 step)
        distance = abs(agent_pos[0] - door[0]) + abs(agent_pos[1] - door[1])
        return distance <= 1

    def reward(self, reward):
        self.steps_taken += 1

        # Goal reached - big bonus!
        if reward > 0:
            return self.goal_bonus

        total_reward = 0.0
        agent_tuple = tuple(self.env.agent_pos)
        current_room = self._get_room(self.env.agent_pos)
        goal_room = self._get_goal_room()

        # ============================================
        # 1. WAYPOINT PROGRESS (most important!)
        # ============================================

        # current_distance = self._compute_distance_to_target()
        # distance_change = self.previous_distance - current_distance
        # self.previous_distance = current_distance

        current_dy, current_dx = self._compute_distance_to_target()

        # Check if reached door (waypoint 1)
        if not self.reached_door and self._is_at_door():
            total_reward += 0.25  # Bonus for reaching door
            self.reached_door = True

        # Check if entered correct room (waypoint 2)
        if current_room != self.previous_room:
            if current_room == goal_room:
                if not self.entered_correct_room:
                    total_reward += 0.3  # Big bonus for entering goal room
                    self.entered_correct_room = True
                    # Reset distance tracking to now target the goal
                    self.previous_distance = self._compute_distance_to_target()
                else:
                    total_reward += 0.1
                self.corridor_counter = 0
            elif current_room == "corridor" and self.previous_room == goal_room:
                # Left the goal room - penalty
                total_reward -= 0.2
            elif current_room != "corridor":
                # Entered wrong room
                total_reward -= 0.15

        self.previous_room = current_room

        # # ============================================
        # # 2. DISTANCE REWARD (toward current target)
        # # ============================================

        # urgency_factor = max(0.5, 1.0 - (self.steps_taken / self.env.max_steps) * 0.5)

        # if distance_change > 0:
        #     # Moving closer to target - reward
        #     total_reward += distance_change * self.distance_scale * urgency_factor
        #     self.stuck_counter = 0
        # elif distance_change < 0:
        #     # Moving away from target - penalty
        #     total_reward += distance_change * self.distance_scale * 1.5
        #     self.stuck_counter = 0
        # else:
        #     # Not moving - ESCALATING penalty
        #     self.stuck_counter += 1
        #     escalating_penalty = -0.01 * (1 + self.stuck_counter * 0.15)
        #     total_reward += escalating_penalty

        # ============================================
        # 2. DISTANCE REWARD (X and Y separately)
        # ============================================

        current_dy, current_dx = self._compute_distance_to_target()
        prev_dy, prev_dx = self.previous_distance
        self.previous_distance = (current_dy, current_dx)

        urgency_factor = max(0.5, 1.0 - (self.steps_taken / self.env.max_steps) * 0.5)

        # Calculate change in each axis
        y_change = prev_dy - current_dy  # Positive = got closer in Y
        x_change = prev_dx - current_dx  # Positive = got closer in X

        # Reward/penalize each axis independently
        y_reward = 0.0
        x_reward = 0.0

        if y_change > 0:
            y_reward = y_change * self.distance_scale * urgency_factor
        elif y_change < 0:
            y_reward = y_change * self.distance_scale * 1.5  # Harsher penalty

        if x_change > 0:
            x_reward = x_change * self.distance_scale * urgency_factor
        elif x_change < 0:
            x_reward = x_change * self.distance_scale * 1.5  # Harsher penalty

        total_reward += y_reward + x_reward

        # Bonus for making progress on BOTH axes simultaneously
        if y_change > 0 and x_change > 0:
            total_reward += 0.05  # Bonus for diagonal progress

        # Track if stuck (no progress on either axis)
        if y_change == 0 and x_change == 0:
            self.stuck_counter += 1
            escalating_penalty = -0.01 * (1 + self.stuck_counter * 0.15)
            total_reward += escalating_penalty
        else:
            self.stuck_counter = 0

        # Extra: penalize being far on one axis while close on another
        # This encourages balanced approach
        total_distance = current_dy + current_dx
        if total_distance > 0:
            imbalance = abs(current_dy - current_dx) / total_distance
            if imbalance > 0.7:  # Very imbalanced (e.g., 9,1 or 1,9)
                total_reward -= 0.01  # Small penalty for imbalance

        # ============================================
        # 3. CORRIDOR PENALTY (escalating)
        # ============================================

        if current_room == "corridor":
            self.corridor_counter += 1
            corridor_penalty = -0.005 * (1 + self.corridor_counter * 0.05)
            total_reward += corridor_penalty
        else:
            self.corridor_counter = 0

        # ============================================
        # 4. EXPLORATION BONUS
        # ============================================

        if agent_tuple not in self.visited:
            if current_room == goal_room:
                total_reward += 0.08
            elif current_room == "corridor":
                total_reward += 0.03
            else:
                total_reward += 0.01
            self.visited.add(agent_tuple)

        # ============================================
        # 5. DIRECTION HINT
        # ============================================
        # Small bonus for facing toward the target

        dir_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        agent_dir = self.env.agent_dir
        dy, dx = dir_vectors[agent_dir]

        # Where is the target relative to agent?
        if not self.entered_correct_room:
            target = self.door_position
        else:
            target = self.env.goal_pos

        target_dy = target[0] - self.env.agent_pos[0]
        target_dx = target[1] - self.env.agent_pos[1]

        # Reward if facing roughly toward target
        facing_reward = 0.0
        if target_dx > 0 and dx > 0:  # Target is right, facing right
            facing_reward += 0.005
        if target_dx < 0 and dx < 0:  # Target is left, facing left
            facing_reward += 0.005
        if target_dy > 0 and dy > 0:  # Target is down, facing down
            facing_reward += 0.005
        if target_dy < 0 and dy < 0:  # Target is up, facing up
            facing_reward += 0.005

        total_reward += facing_reward

        # ============================================
        # 6. BASE STEP COST
        # ============================================

        total_reward -= 0.001

        return total_reward


# class EscalatingRewardWrapper(gym.RewardWrapper):
#     """
#     R5: AGGRESSIVE Reward Wrapper

#     Key insight: The agent needs to TURN, not just go forward.
#     We need to make hitting walls EXTREMELY painful and turning REWARDING.
#     """

#     def __init__(self, env, distance_scale: float = 0.1, goal_bonus: float = 10.0):
#         super().__init__(env)
#         self.reward_type = "R5_escalating"
#         self.distance_scale = distance_scale
#         self.goal_bonus = goal_bonus

#         # State tracking
#         self.previous_distance = None
#         self.previous_pos = None
#         self.wall_hits = 0
#         self.steps_since_progress = 0
#         self.visited = set()
#         self.steps_taken = 0
#         self.entered_correct_room = False
#         self.door_position = None

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)

#         self.previous_pos = tuple(self.env.agent_pos)
#         self.wall_hits = 0
#         self.steps_since_progress = 0
#         self.visited = set()
#         self.visited.add(self.previous_pos)
#         self.steps_taken = 0
#         self.entered_correct_room = False

#         # Find door to goal room
#         self.door_position = self._get_goal_room_door()
#         self.previous_distance = self._get_distance_to_target()

#         return obs, info

#     def _get_room(self, pos) -> str:
#         y, x = pos[0], pos[1]
#         if 7 <= x <= 11:
#             return "corridor"
#         if x < 7:
#             if y < 6:
#                 return "left_top"
#             elif y < 12:
#                 return "left_middle"
#             else:
#                 return "left_bottom"
#         if x > 11:
#             if y < 6:
#                 return "right_top"
#             elif y < 12:
#                 return "right_middle"
#             else:
#                 return "right_bottom"
#         return "unknown"

#     def _get_goal_room(self) -> str:
#         return self._get_room(list(self.env.goal_pos))

#     def _get_goal_room_door(self) -> tuple:
#         goal_room = self._get_goal_room()
#         door_map = {
#             "left_top": (3, 7),
#             "left_middle": (9, 7),
#             "left_bottom": (15, 7),
#             "right_top": (3, 11),
#             "right_middle": (9, 11),
#             "right_bottom": (15, 11),
#         }
#         return door_map.get(goal_room, (9, 9))

#     def _get_distance_to_target(self) -> tuple:
#         """Returns (dy, dx) to current target."""
#         agent_pos = self.env.agent_pos

#         if not self.entered_correct_room:
#             target = self.door_position
#         else:
#             target = self.env.goal_pos

#         dy = abs(agent_pos[0] - target[0])
#         dx = abs(agent_pos[1] - target[1])
#         return (dy, dx)

#     def step(self, action):
#         """Override step to track what actually happened."""

#         old_pos = tuple(self.env.agent_pos)
#         old_dir = self.env.agent_dir

#         obs, original_reward, terminated, truncated, info = self.env.step(action)

#         new_pos = tuple(self.env.agent_pos)
#         new_dir = self.env.agent_dir

#         self.steps_taken += 1

#         # =============================================
#         # GOAL REACHED - MASSIVE REWARD
#         # =============================================
#         if original_reward > 0:
#             return obs, self.goal_bonus, terminated, truncated, info

#         total_reward = 0.0

#         # =============================================
#         # 1. WALL HIT DETECTION - BRUTAL PENALTY
#         # =============================================
#         tried_to_move = action == 2  # FORWARD action
#         actually_moved = old_pos != new_pos

#         if tried_to_move and not actually_moved:
#             # HIT A WALL! Escalating punishment
#             self.wall_hits += 1
#             wall_penalty = -0.1 * (1 + self.wall_hits * 0.5)  # Gets worse fast
#             total_reward += wall_penalty
#             self.steps_since_progress += 1

#         # =============================================
#         # 2. TURNING REWARD - Encourage turning!
#         # =============================================
#         turned = action == 0 or action == 1  # LEFT or RIGHT

#         if turned:
#             # Check if turn was toward the target
#             target = self.door_position if not self.entered_correct_room else self.env.goal_pos

#             agent_y, agent_x = new_pos
#             target_y, target_x = target

#             # Direction vectors: 0=right, 1=down, 2=left, 3=up
#             # Check if now facing toward target
#             facing_good = False

#             if new_dir == 0 and target_x > agent_x:  # Facing right, target is right
#                 facing_good = True
#             elif new_dir == 2 and target_x < agent_x:  # Facing left, target is left
#                 facing_good = True
#             elif new_dir == 1 and target_y > agent_y:  # Facing down, target is below
#                 facing_good = True
#             elif new_dir == 3 and target_y < agent_y:  # Facing up, target is above
#                 facing_good = True

#             if facing_good:
#                 total_reward += 0.05  # Reward for turning toward target
#                 self.wall_hits = 0  # Reset wall hit counter
#             else:
#                 total_reward += 0.01  # Small reward for any turn (exploration)

#         # =============================================
#         # 3. MOVEMENT REWARD - X and Y separately
#         # =============================================
#         if actually_moved:
#             current_dy, current_dx = self._get_distance_to_target()
#             prev_dy, prev_dx = self.previous_distance

#             y_change = prev_dy - current_dy
#             x_change = prev_dx - current_dx

#             # Reward each axis
#             if y_change > 0:
#                 total_reward += 0.15  # Good Y progress
#             elif y_change < 0:
#                 total_reward -= 0.1  # Bad Y progress

#             if x_change > 0:
#                 total_reward += 0.15  # Good X progress
#             elif x_change < 0:
#                 total_reward -= 0.1  # Bad X progress

#             # BONUS for progress on both axes
#             if y_change > 0 and x_change > 0:
#                 total_reward += 0.1

#             # Update tracking
#             self.previous_distance = (current_dy, current_dx)
#             self.wall_hits = 0  # Reset wall counter on successful move
#             self.steps_since_progress = (
#                 0 if (y_change > 0 or x_change > 0) else self.steps_since_progress + 1
#             )

#             # New cell bonus
#             if new_pos not in self.visited:
#                 total_reward += 0.1
#                 self.visited.add(new_pos)

#         # =============================================
#         # 4. ROOM TRANSITIONS
#         # =============================================
#         old_room = self._get_room(old_pos)
#         new_room = self._get_room(new_pos)
#         goal_room = self._get_goal_room()

#         if old_room != new_room:
#             if new_room == goal_room:
#                 if not self.entered_correct_room:
#                     total_reward += 1.0  # BIG bonus for entering goal room
#                     self.entered_correct_room = True
#                     self.previous_distance = self._get_distance_to_target()
#             elif new_room != "corridor" and new_room != goal_room:
#                 total_reward -= 0.5  # Penalty for wrong room

#         # =============================================
#         # 5. STUCK PENALTY - Escalating
#         # =============================================
#         if self.steps_since_progress > 5:
#             stuck_penalty = -0.05 * (self.steps_since_progress - 5)
#             total_reward += stuck_penalty

#         # =============================================
#         # 6. TIME PRESSURE
#         # =============================================
#         total_reward -= 0.005  # Constant small cost per step

#         self.previous_pos = new_pos

#         return obs, total_reward, terminated, truncated, info

#     def reward(self, reward):
#         # Not used - we override step() instead
#         return reward


class HierarchicalRewardWrapper(gym.RewardWrapper):
    """
    R6: Hierarchical Reward - Clear priorities

    Priority 1: Reach the correct room
    Priority 2: Reach the goal (once in room)

    Simple penalties for clearly bad behavior.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_type = "R6_hierarchical"

        # State
        self.in_goal_room = False
        self.steps_in_corridor = 0
        self.steps_in_wrong_room = 0
        self.consecutive_wall_hits = 0
        self.previous_pos = None
        self.door_position = None
        self.visited = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.in_goal_room = False
        self.steps_in_corridor = 0
        self.steps_in_wrong_room = 0
        self.consecutive_wall_hits = 0
        self.previous_pos = tuple(self.env.agent_pos)
        self.door_position = self._get_door_for_goal()
        self.visited = set()
        self.visited.add(self.previous_pos)

        return obs, info

    def _get_room(self, pos) -> str:
        y, x = pos[0], pos[1]
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

    def _get_goal_room(self) -> str:
        return self._get_room(list(self.env.goal_pos))

    def _get_door_for_goal(self) -> tuple:
        goal_room = self._get_goal_room()
        doors = {
            "left_top": (3, 7),
            "left_middle": (9, 7),
            "left_bottom": (15, 7),
            "right_top": (3, 11),
            "right_middle": (9, 11),
            "right_bottom": (15, 11),
        }
        return doors.get(goal_room, (9, 7))

    def _distance(self, pos1, pos2) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        current_room = self._get_room(new_pos)
        goal_room = self._get_goal_room()

        # ===================
        # GOAL REACHED
        # ===================
        if base_reward > 0:
            return obs, 10.0, terminated, truncated, info

        reward = 0.0

        # ===================
        # PHASE 1: NOT IN GOAL ROOM YET
        # ===================
        if not self.in_goal_room:

            # Check if just entered goal room
            if current_room == goal_room:
                self.in_goal_room = True
                reward += 2.0  # Big bonus
                self.steps_in_corridor = 0
                self.steps_in_wrong_room = 0
            else:
                # Reward for moving toward the door
                old_dist_to_door = self._distance(old_pos, self.door_position)
                new_dist_to_door = self._distance(new_pos, self.door_position)

                if new_dist_to_door < old_dist_to_door:
                    reward += 0.1  # Getting closer to door
                elif new_dist_to_door > old_dist_to_door:
                    reward -= 0.05  # Moving away from door

                # Track bad locations
                if current_room == "corridor":
                    self.steps_in_corridor += 1
                    self.steps_in_wrong_room = 0
                    if self.steps_in_corridor > 20:
                        reward -= 0.01 * (self.steps_in_corridor - 20)
                elif current_room != goal_room:
                    self.steps_in_wrong_room += 1
                    self.steps_in_corridor = 0
                    reward -= 0.02 * self.steps_in_wrong_room  # Wrong room hurts more

        # ===================
        # PHASE 2: IN GOAL ROOM
        # ===================
        else:
            goal_pos = self.env.goal_pos
            old_dist_to_goal = self._distance(old_pos, goal_pos)
            new_dist_to_goal = self._distance(new_pos, goal_pos)

            if new_dist_to_goal < old_dist_to_goal:
                reward += 0.2  # Strong reward for approaching goal
            elif new_dist_to_goal > old_dist_to_goal:
                reward -= 0.1

            # Penalty for leaving goal room
            if current_room != goal_room:
                reward -= 1.0
                self.in_goal_room = False

        # ===================
        # WALL HIT PENALTY
        # ===================
        tried_forward = action == 2
        moved = old_pos != new_pos

        if tried_forward and not moved:
            self.consecutive_wall_hits += 1
            reward -= 0.002 * self.consecutive_wall_hits
        else:
            self.consecutive_wall_hits = 0

        # ===================
        # EXPLORATION BONUS
        # ===================
        if new_pos not in self.visited:
            self.visited.add(new_pos)

            if current_room == goal_room:
                reward += 0.15  # High bonus for exploring goal room
            elif current_room == "corridor":
                reward += 0.05  # Medium bonus for corridor (need to traverse it)
            else:
                reward += 0.02  # Small bonus for wrong rooms (still exploring)

        # ===================
        # TURNING REWARD
        # ===================
        turned = action == 0 or action == 1  # LEFT or RIGHT

        if turned:
            agent_dir = self.env.agent_dir
            agent_y, agent_x = new_pos

            if not self.in_goal_room:
                target_y, target_x = self.door_position
            else:
                target_y, target_x = self.env.goal_pos

            diff_y = target_y - agent_y
            diff_x = target_x - agent_x

            # Check if now facing toward target
            good_turn = False
            if agent_dir == 0 and diff_x > 0:  # Facing right, target is right
                good_turn = True
            elif agent_dir == 2 and diff_x < 0:  # Facing left, target is left
                good_turn = True
            elif agent_dir == 1 and diff_y > 0:  # Facing down, target is down
                good_turn = True
            elif agent_dir == 3 and diff_y < 0:  # Facing up, target is up
                good_turn = True

            if good_turn:
                reward += 0.1  # Significant reward for turning toward target

        # ===================
        # SMALL STEP COST
        # ===================
        reward -= 0.001

        self.previous_pos = new_pos

        return obs, reward, terminated, truncated, info

    def reward(self, reward):
        return reward


class SimpleRewardWrapper(gym.RewardWrapper):
    """
    R7: Simple and Clean

    + Distance to target decreases
    + Turning toward target (once per move)
    + Entering goal room
    + Reaching goal (huge)
    - Each step costs a little
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_type = "R7_simple"
        self.previous_distance = None
        self.in_goal_room = False
        self.facing_bonus_available = True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._distance_to_goal()
        self.in_goal_room = False
        self.facing_bonus_available = True
        return obs, info

    def _distance_to_goal(self):
        pos = self.env.agent_pos
        goal = self.env.goal_pos
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _in_goal_room(self):
        y, x = self.env.agent_pos
        gy, gx = self.env.goal_pos
        # Same room if same side of walls
        if x < 7 and gx < 7:
            if (y < 6 and gy < 6) or (6 <= y < 12 and 6 <= gy < 12) or (y >= 12 and gy >= 12):
                return True
        return False

    def _facing_target(self):
        pos = self.env.agent_pos
        goal = self.env.goal_pos
        direction = self.env.agent_dir

        dy = goal[0] - pos[0]
        dx = goal[1] - pos[1]

        # 0=right, 1=down, 2=left, 3=up
        if direction == 0 and dx > 0:
            return True
        if direction == 2 and dx < 0:
            return True
        if direction == 3 and dy < 0:
            return True
        if direction == 1 and dy > 0:
            return True
        return False

    def step(self, action):
        old_pos = tuple(self.env.agent_pos)

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        new_pos = tuple(self.env.agent_pos)
        moved = old_pos != new_pos

        # Goal reached = huge reward
        if base_reward > 0:
            return obs, 100.0, terminated, truncated, info

        reward = 0.0

        # 1. Distance reward
        current_distance = self._distance_to_goal()
        if current_distance < self.previous_distance:
            reward += 1.0  # Got closer
        elif current_distance > self.previous_distance:
            reward -= 0.5  # Got further
        self.previous_distance = current_distance

        # 2. Turning toward target (once until move)
        if action in [0, 1]:  # Turned
            if self._facing_target() and self.facing_bonus_available:
                reward += 0.5
                self.facing_bonus_available = False

        if moved:
            self.facing_bonus_available = True  # Reset after moving

        # 3. Entering goal room
        now_in_goal_room = self._in_goal_room()
        if now_in_goal_room and not self.in_goal_room:
            reward += 10.0  # First time entering
        self.in_goal_room = now_in_goal_room

        # 4. Step cost
        reward -= 0.1

        return obs, reward, terminated, truncated, info

    def reward(self, reward):
        return reward


# =============================================================================
# REWARD WRAPPER FACTORY
# =============================================================================

REWARD_WRAPPERS = {
    "R1": SparseRewardWrapper,
    "R2": DistanceRewardWrapper,
    "R3": StepPenaltyRewardWrapper,
    "R4": CombinedRewardWrapper,
    "R5": EscalatingRewardWrapper,
    "R6": HierarchicalRewardWrapper,
    "R7": SimpleRewardWrapper,
}


def apply_reward_wrapper(env, reward_type: str, **kwargs):
    """Apply a reward wrapper to an environment"""
    if reward_type not in REWARD_WRAPPERS:
        raise ValueError(
            f"Unknown reward type: {reward_type}. Available: {list(REWARD_WRAPPERS.keys())}"
        )

    wrapper_class = REWARD_WRAPPERS[reward_type]
    return wrapper_class(env, **kwargs)


def list_reward_types():
    """Print available reward types"""
    print("Available Reward Strategies:")
    print("=" * 60)
    descriptions = {
        "R1": "Sparse - +1 at goal only (hardest to learn)",
        "R2": "Distance-based - reward for getting closer",
        "R3": "Step penalty - efficiency bonus at goal",
        "R4": "Combined - distance + goal + step penalty",
        "R5": "Escalating - increasing penalty for staying stuck",
        "R6": "Hierarchical - room first, then goal",
        "R7": "Keep it simple",
    }
    for reward_id, desc in descriptions.items():
        print(f"  {reward_id}: {desc}")


if __name__ == "__main__":
    list_reward_types()
