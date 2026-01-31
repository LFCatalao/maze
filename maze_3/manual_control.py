"""
Manual control script for testing the omnidirectional movement environment
with fog of war visualization.

Controls:
- Arrow keys: Move Up/Down/Left/Right
- R: Reset environment
- ESC: Quit
"""

import pygame
import numpy as np
import sys
from environments.base_env import LockedRoomEnv, Objects, COLOR_MAP, Colors, Actions
from environments.reward_wrappers import SimpleRewardWrapper


def create_test_env_6():
    """
    Create Environment 6 - E6_MULTI_DOOR_RANDOM
    Fixed agent, random exit behind random door with key chain (5 doors)
    """
    env = LockedRoomEnv(
        size=19,
        observation_mode="partial_view",  # Will be forced to 7x7
        render_mode=None,  # We'll handle rendering manually
        max_steps=500,
        agent_view_size=3,  # 7x7 view (2*3+1)
        # E6_MULTI_DOOR_RANDOM configuration
        fixed_agent_pos=(9, 9),  # Center of corridor
        fixed_goal_pos=None,  # Random goal
        num_doors=4,
        include_key=True,
        locked_door=True,
        randomize_doors=True,
        goal_in_locked_room=True,
        enable_key_chain=True,
        verbose=True,
    )
    # Wrap with reward shaping
    env = SimpleRewardWrapper(env)
    return env


def render_dual_view(screen, env, obs, cell_size=32, last_reward=0.0):
    """
    Render both the explored map (with fog of war) and the 7x7 partial view.
    
    Args:
        screen: Pygame screen
        env: Environment instance
        obs: Current observation
        cell_size: Size of each cell in pixels
    """
    screen.fill((0, 0, 0))
    
    # Calculate layout
    full_map_size = env.size * cell_size
    partial_view_size = 7 * cell_size
    padding = 20
    
    # Draw explored map (left side)
    explored_map = obs['explored_map']
    for y in range(env.size):
        for x in range(env.size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            
            obj = explored_map[y, x, 0]
            color_idx = explored_map[y, x, 1]
            state = explored_map[y, x, 2]
            
            # A cell is unexplored if all values are 0
            # Explored empty cells have state=1 (marked in _update_explored_map)
            is_unexplored = (obj == 0 and color_idx == 0 and state == 0)
            
            if is_unexplored:
                # Unexplored: pure black
                pygame.draw.rect(screen, (0, 0, 0), rect)
            elif obj == Objects.WALL:
                # Explored wall: dark grey
                pygame.draw.rect(screen, (64, 64, 64), rect)
            elif obj == Objects.DOOR:
                color = COLOR_MAP.get(color_idx, (128, 128, 128))
                pygame.draw.rect(screen, color, rect)
                if state == 2:  # Locked
                    pygame.draw.circle(screen, (0, 0, 0), rect.center, 5)
                elif state == 1:  # Closed
                    pygame.draw.rect(screen, (0, 0, 0), rect, 2)
            elif obj == Objects.KEY:
                pygame.draw.rect(screen, (255, 255, 255), rect)  # White background
                color = COLOR_MAP.get(color_idx, (128, 128, 128))
                pygame.draw.circle(screen, color, rect.center, 8)
            elif obj == Objects.GOAL:
                pygame.draw.rect(screen, (0, 255, 0), rect)
            else:  # Empty explored cell (obj == 0 but state == 1)
                # Explored empty: white
                pygame.draw.rect(screen, (255, 255, 255), rect)
            
            # Draw grid lines
            pygame.draw.rect(screen, (100, 100, 100), rect, 1)
    
    # Draw agent position on explored map
    agent_y, agent_x = env.agent_pos
    agent_rect = pygame.Rect(
        agent_x * cell_size,
        agent_y * cell_size,
        cell_size,
        cell_size
    )
    pygame.draw.circle(screen, (255, 0, 0), agent_rect.center, cell_size // 3)
    
    # Draw carrying indicator on explored map
    if env.carrying is not None:
        carry_color = COLOR_MAP.get(env.carrying, (128, 128, 128))
        pygame.draw.circle(screen, carry_color, (agent_rect.centerx, agent_rect.top - 5), 5)
    
    # Draw 7x7 partial view (right side)
    partial_view = obs['image']
    offset_x = full_map_size + padding
    offset_y = 50
    
    for i in range(7):
        for j in range(7):
            rect = pygame.Rect(
                offset_x + j * cell_size,
                offset_y + i * cell_size,
                cell_size,
                cell_size
            )
            
            obj = partial_view[i, j, 0]
            color_idx = partial_view[i, j, 1]
            state = partial_view[i, j, 2]
            
            if obj == Objects.WALL:
                pygame.draw.rect(screen, (64, 64, 64), rect)
            elif obj == Objects.DOOR:
                color = COLOR_MAP.get(color_idx, (128, 128, 128))
                pygame.draw.rect(screen, color, rect)
                if state == 2:
                    pygame.draw.circle(screen, (0, 0, 0), rect.center, 5)
                elif state == 1:
                    pygame.draw.rect(screen, (0, 0, 0), rect, 2)
            elif obj == Objects.KEY:
                pygame.draw.rect(screen, (240, 240, 240), rect)
                color = COLOR_MAP.get(color_idx, (128, 128, 128))
                pygame.draw.circle(screen, color, rect.center, 8)
            elif obj == Objects.GOAL:
                pygame.draw.rect(screen, (0, 255, 0), rect)
            else:
                pygame.draw.rect(screen, (240, 240, 240), rect)
            
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
    
    # Highlight center cell (agent position in partial view)
    center_rect = pygame.Rect(
        offset_x + 3 * cell_size,
        offset_y + 3 * cell_size,
        cell_size,
        cell_size
    )
    pygame.draw.circle(screen, (255, 0, 0), center_rect.center, cell_size // 3)
    
    # Draw text labels
    font = pygame.font.Font(None, 24)
    
    # Title for explored map
    text = font.render("Explored Map (Fog of War)", True, (255, 255, 255))
    screen.blit(text, (10, full_map_size + 10))
    
    # Title for partial view
    text = font.render("Agent's 7x7 View", True, (255, 255, 255))
    screen.blit(text, (offset_x, offset_y - 30))
    
    # Info text
    info_y = offset_y + partial_view_size + 20
    text = font.render(f"Steps: {env.step_count}/{env.max_steps}", True, (255, 255, 255))
    screen.blit(text, (offset_x, info_y))
    
    carrying_text = "nothing"
    if env.carrying is not None:
        from environments.base_env import COLOR_NAMES
        carrying_text = COLOR_NAMES.get(env.carrying, "unknown") + " key"
    text = font.render(f"Carrying: {carrying_text}", True, (255, 255, 255))
    screen.blit(text, (offset_x, info_y + 30))
    
    text = font.render(f"Agent: ({agent_y}, {agent_x})", True, (255, 255, 255))
    screen.blit(text, (offset_x, info_y + 60))
    
    # Reward display
    reward_color = (0, 255, 0) if last_reward > 0 else (255, 255, 255)
    text = font.render(f"Last Reward: {last_reward:.1f}", True, reward_color)
    screen.blit(text, (offset_x, info_y + 90))
    
    # Controls
    controls = [
        "Controls:",
        "Arrow Keys: Move",
        "R: Reset",
        "ESC: Quit"
    ]
    for i, line in enumerate(controls):
        text = font.render(line, True, (200, 200, 200))
        screen.blit(text, (offset_x, info_y + 130 + i * 25))
    
    pygame.display.flip()


def get_base_env(env):
    """Get the base environment from a wrapped environment"""
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    return base_env


def main():
    """Main loop for manual control"""
    # Initialize pygame
    pygame.init()
    
    # Create environment
    print("Creating Environment 6...")
    env = create_test_env_6()
    obs, info = env.reset()
    
    # Get base environment for accessing attributes
    base_env = get_base_env(env)
    
    # Setup display
    cell_size = 32
    full_map_size = base_env.size * cell_size
    partial_view_size = 7 * cell_size
    padding = 20
    
    screen_width = full_map_size + padding + partial_view_size + padding
    screen_height = full_map_size + 50
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Locked Room Environment - Manual Control")
    clock = pygame.time.Clock()
    
    print("\nEnvironment created!")
    print(f"Mission: {base_env.mission}")
    print(f"Agent starts at: {base_env.agent_pos}")
    print(f"Goal is at: {base_env.goal_pos}")
    print("\nUse arrow keys to move, R to reset, ESC to quit")
    
    running = True
    terminated = False
    truncated = False
    last_reward = 0.0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                action = None
                
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_r:
                    # Reset environment
                    print("\nResetting environment...")
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                    last_reward = 0.0
                    print(f"Agent reset to: {base_env.agent_pos}")
                
                elif not terminated and not truncated:
                    # Movement controls
                    if event.key == pygame.K_UP:
                        action = Actions.UP
                    elif event.key == pygame.K_DOWN:
                        action = Actions.DOWN
                    elif event.key == pygame.K_LEFT:
                        action = Actions.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Actions.RIGHT
                    
                    # Execute action
                    if action is not None:
                        obs, reward, terminated, truncated, info = env.step(action)
                        last_reward = reward
                        
                        # Print reward for each step
                        print(f"Step {base_env.step_count}: Reward = {reward:.2f}")
                        
                        if terminated:
                            print(f"\n*** GOAL REACHED! Total Reward: {reward:.2f} ***")
                            print(f"Completed in {base_env.step_count} steps")
                        
                        if truncated:
                            print(f"\nEpisode truncated (max steps reached)")
        
        # Render
        render_dual_view(screen, base_env, obs, cell_size, last_reward)
        clock.tick(30)  # 30 FPS
    
    pygame.quit()
    print("\nExiting...")


if __name__ == "__main__":
    main()
