#!/usr/bin/env python3
"""
Debug Script with A* Pathfinding

This script helps you:
1. See the optimal path (A*) to the goal
2. Debug reward values step-by-step
3. Compare agent behavior to optimal behavior
"""

import sys
import os
import time
import heapq
from typing import List, Tuple, Set, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments import LockedRoomEnv, get_config, apply_reward_wrapper, Objects


# =============================================================================
# A* PATHFINDING
# =============================================================================


def get_neighbors(pos: Tuple[int, int], grid, size: int) -> List[Tuple[int, int]]:
    """Get valid neighboring positions (not walls, not locked doors)."""
    y, x = pos
    neighbors = []

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx

        if 0 <= ny < size and 0 <= nx < size:
            obj = grid[ny, nx, 0]

            # Can walk through empty, goal, key, and open doors
            if obj == Objects.EMPTY or obj == Objects.GOAL or obj == Objects.KEY:
                neighbors.append((ny, nx))
            elif obj == Objects.DOOR:
                state = grid[ny, nx, 2]
                if state == 0:  # Open door
                    neighbors.append((ny, nx))

    return neighbors


def heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def a_star(
    start: Tuple[int, int], goal: Tuple[int, int], grid, size: int
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm.

    Returns list of positions from start to goal, or None if no path exists.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0}
    f_score: Dict[Tuple[int, int], float] = {start: heuristic(start, goal)}

    open_set_hash: Set[Tuple[int, int]] = {start}

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, grid, size):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return None  # No path found


def get_action_for_move(
    from_pos: Tuple[int, int], to_pos: Tuple[int, int], current_dir: int
) -> List[int]:
    """
    Get the sequence of actions needed to move from from_pos to to_pos.

    Returns list of actions (may need to turn first, then move forward).
    """
    dy = to_pos[0] - from_pos[0]
    dx = to_pos[1] - from_pos[1]

    # Determine required direction
    # 0=right, 1=down, 2=left, 3=up
    if dx > 0:
        required_dir = 0  # right
    elif dx < 0:
        required_dir = 2  # left
    elif dy > 0:
        required_dir = 1  # down
    else:
        required_dir = 3  # up

    actions = []

    # Turn to face the right direction
    while current_dir != required_dir:
        # Decide whether to turn left or right
        diff = (required_dir - current_dir) % 4
        if diff == 1 or diff == -3:
            actions.append(1)  # Turn right
            current_dir = (current_dir + 1) % 4
        else:
            actions.append(0)  # Turn left
            current_dir = (current_dir - 1) % 4

    # Move forward
    actions.append(2)

    return actions, required_dir


# =============================================================================
# DEBUG FUNCTIONS
# =============================================================================


def print_grid_with_path(env, path: List[Tuple[int, int]]):
    """Print ASCII grid with path marked."""
    grid = env.grid
    size = env.size
    agent_pos = tuple(env.agent_pos)
    goal_pos = env.goal_pos

    path_set = set(path)

    print("\nGrid with A* path (marked with *):")
    print("=" * (size + 2))

    for y in range(size):
        row = ""
        for x in range(size):
            pos = (y, x)
            obj = grid[y, x, 0]

            if pos == agent_pos:
                row += "A"
            elif pos == goal_pos:
                row += "G"
            elif pos in path_set:
                row += "*"
            elif obj == Objects.WALL:
                row += "#"
            elif obj == Objects.DOOR:
                state = grid[y, x, 2]
                row += "D" if state > 0 else "."
            elif obj == Objects.KEY:
                row += "K"
            else:
                row += "."
        print(row)

    print("=" * (size + 2))


def run_astar_agent(
    env_id: str = "E1", reward_id: str = "R6", render: bool = True, delay_ms: int = 200
):
    """
    Run A* optimal agent and show rewards at each step.
    """
    print("=" * 70)
    print("A* OPTIMAL AGENT")
    print("=" * 70)

    config = get_config(env_id)

    env = LockedRoomEnv(
        size=config.size,
        observation_mode="full_map",
        render_mode="human" if render else None,
        max_steps=config.max_steps,
        fixed_agent_pos=config.fixed_agent_pos,
        fixed_goal_pos=config.fixed_goal_pos,
        num_doors=config.num_doors,
        randomize_doors=config.randomize_doors,
        include_key=config.include_key,
        locked_door=config.locked_door,
        verbose=False,
    )

    env = apply_reward_wrapper(env, reward_id)

    obs, info = env.reset()

    start = tuple(env.env.agent_pos)  # env.env because of wrapper
    goal = env.env.goal_pos

    print(f"\nStart: {start}")
    print(f"Goal: {goal}")
    print(f"Reward type: {reward_id}")

    # Find optimal path
    path = a_star(start, goal, env.env.grid, env.env.size)

    if path is None:
        print("\nERROR: No path found! Check if goal is reachable.")
        env.close()
        return

    print(f"Optimal path length: {len(path) - 1} steps")
    print(f"Path: {path[:5]}...{path[-3:]}" if len(path) > 8 else f"Path: {path}")

    print_grid_with_path(env.env, path)

    # Execute path
    print("\n" + "=" * 70)
    print("EXECUTING OPTIMAL PATH")
    print("=" * 70)

    current_dir = env.env.agent_dir
    total_reward = 0
    step_count = 0

    action_names = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "PICKUP", 4: "TOGGLE"}

    for i in range(len(path) - 1):
        from_pos = path[i]
        to_pos = path[i + 1]

        actions, current_dir = get_action_for_move(from_pos, to_pos, current_dir)

        for action in actions:
            if render:
                env.render()
                time.sleep(delay_ms / 1000)

            old_pos = tuple(env.env.agent_pos)
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.env.agent_pos)

            total_reward += reward
            step_count += 1

            # Print reward breakdown
            moved = "MOVED" if old_pos != new_pos else "no move"
            print(
                f"Step {step_count:3d}: {action_names[action]:8s} | {old_pos} -> {new_pos} | reward: {reward:+.4f} | total: {total_reward:+.4f} | {moved}"
            )

            if terminated:
                print("\n" + "=" * 70)
                print(f"GOAL REACHED!")
                print(f"Total steps: {step_count}")
                print(f"Total reward: {total_reward:.4f}")
                print("=" * 70)

                if render:
                    env.render()
                    time.sleep(1)

                env.close()
                return

    print("\nPath completed but goal not reached?")
    env.close()


def debug_rewards_manual(env_id: str = "E1", reward_id: str = "R6"):
    """
    Manual control with detailed reward breakdown.
    """
    import pygame

    print("=" * 70)
    print("MANUAL DEBUG MODE")
    print("=" * 70)
    print("\nControls:")
    print("  ← → : Turn left/right")
    print("  ↑   : Move forward")
    print("  SPACE: Pick up")
    print("  E   : Toggle door")
    print("  R   : Reset")
    print("  P   : Print A* path")
    print("  ESC : Quit")
    print("=" * 70)

    config = get_config(env_id)

    env = LockedRoomEnv(
        size=config.size,
        observation_mode="full_map",
        render_mode="human",
        max_steps=config.max_steps,
        fixed_agent_pos=config.fixed_agent_pos,
        fixed_goal_pos=config.fixed_goal_pos,
        num_doors=config.num_doors,
        randomize_doors=config.randomize_doors,
        include_key=config.include_key,
        locked_door=config.locked_door,
        verbose=False,
    )

    env = apply_reward_wrapper(env, reward_id)

    obs, info = env.reset()

    print(f"\nStart: {env.env.agent_pos}")
    print(f"Goal: {env.env.goal_pos}")

    # Show initial A* path
    path = a_star(tuple(env.env.agent_pos), env.env.goal_pos, env.env.grid, env.env.size)
    if path:
        print(f"A* optimal path length: {len(path) - 1}")

    action_names = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "PICKUP", 4: "TOGGLE"}
    total_reward = 0
    step_count = 0

    running = True
    while running:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                action = None

                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_SPACE:
                    action = 3
                elif event.key == pygame.K_e:
                    action = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    step_count = 0
                    print("\n--- RESET ---")
                    print(f"Start: {env.env.agent_pos}, Goal: {env.env.goal_pos}")
                elif event.key == pygame.K_p:
                    path = a_star(
                        tuple(env.env.agent_pos), env.env.goal_pos, env.env.grid, env.env.size
                    )
                    if path:
                        print_grid_with_path(env.env, path)
                        print(f"Path length: {len(path) - 1}")
                elif event.key == pygame.K_ESCAPE:
                    running = False

                if action is not None:
                    old_pos = tuple(env.env.agent_pos)
                    old_dir = env.env.agent_dir

                    obs, reward, terminated, truncated, info = env.step(action)

                    new_pos = tuple(env.env.agent_pos)
                    new_dir = env.env.agent_dir

                    total_reward += reward
                    step_count += 1

                    # Detailed output
                    moved = "MOVED" if old_pos != new_pos else "no move"
                    turned = "TURNED" if old_dir != new_dir else ""

                    print(f"\nStep {step_count}: {action_names[action]}")
                    print(f"  Position: {old_pos} -> {new_pos} ({moved})")
                    print(f"  Direction: {old_dir} -> {new_dir} {turned}")
                    print(f"  Reward: {reward:+.4f}")
                    print(f"  Total reward: {total_reward:+.4f}")

                    # Show distance to goal
                    dist = abs(new_pos[0] - env.env.goal_pos[0]) + abs(
                        new_pos[1] - env.env.goal_pos[1]
                    )
                    print(f"  Distance to goal: {dist}")

                    if terminated:
                        print("\n" + "=" * 50)
                        print(f"GOAL REACHED in {step_count} steps!")
                        print(f"Total reward: {total_reward:.4f}")
                        print("=" * 50)
                        print("Press R to reset or ESC to quit")

        pygame.time.wait(30)

    env.close()


def analyze_reward_function(env_id: str = "E1", reward_id: str = "R6"):
    """
    Analyze reward function by testing specific scenarios.
    """
    print("=" * 70)
    print(f"REWARD FUNCTION ANALYSIS: {reward_id}")
    print("=" * 70)

    config = get_config(env_id)

    scenarios = [
        ("Move toward door", [0, 0, 2, 2, 2]),  # Turn left twice, forward 3x
        ("Move away from door", [1, 1, 2, 2, 2]),  # Turn right twice, forward 3x
        ("Hit wall repeatedly", [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),  # Forward into wall
        ("Spin in place", [0, 0, 0, 0, 1, 1, 1, 1]),  # Turn left 4x, right 4x
    ]

    for scenario_name, actions in scenarios:
        print(f"\n--- {scenario_name} ---")

        env = LockedRoomEnv(
            size=config.size,
            observation_mode="full_map",
            render_mode=None,
            max_steps=config.max_steps,
            fixed_agent_pos=config.fixed_agent_pos,
            fixed_goal_pos=config.fixed_goal_pos,
            num_doors=config.num_doors,
        )
        env = apply_reward_wrapper(env, reward_id)

        obs, info = env.reset()
        total_reward = 0

        action_names = {0: "L", 1: "R", 2: "F", 3: "P", 4: "T"}

        for action in actions:
            old_pos = tuple(env.env.agent_pos)
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.env.agent_pos)
            total_reward += reward

            moved = "M" if old_pos != new_pos else "."
            print(f"  {action_names[action]} {moved} r={reward:+.3f} (total={total_reward:+.3f})")

            if terminated:
                print("  GOAL!")
                break

        env.close()

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug and A* pathfinding tools")
    parser.add_argument(
        "mode",
        choices=["astar", "manual", "analyze"],
        help="astar: watch optimal agent, manual: control with debug info, analyze: test scenarios",
    )
    parser.add_argument("--env", type=str, default="E1", help="Environment config")
    parser.add_argument("--reward", type=str, default="R6", help="Reward wrapper")
    parser.add_argument("--delay", type=int, default=200, help="Delay in ms (for astar)")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()

    if args.mode == "astar":
        run_astar_agent(args.env, args.reward, not args.no_render, args.delay)
    elif args.mode == "manual":
        debug_rewards_manual(args.env, args.reward)
    elif args.mode == "analyze":
        analyze_reward_function(args.env, args.reward)
