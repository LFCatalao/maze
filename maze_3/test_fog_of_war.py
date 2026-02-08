"""
Test the fog of war rendering in evaluate.py
"""

import sys
sys.path.append('.')

from environments.base_env import LockedRoomEnv
from environments.reward_wrappers import SimpleRewardWrapper, SimpleObs

def test_fog_of_war_rendering():
    """Test that fog of war renders correctly"""
    
    env = LockedRoomEnv(
        size=19,
        observation_mode="full_map",
        render_mode="human",
        fixed_agent_pos=(9, 9),
        fixed_goal_pos=(3, 3),
        num_doors=1,
        defined_doors=[{'door_idx': 0, 'key_pos': (9, 4)}],
        verbose=True
    )
    
    env = SimpleObs(env)
    env = SimpleRewardWrapper(env)
    
    obs, info = env.reset()
    
    print("=" * 60)
    print("FOG OF WAR TEST")
    print("=" * 60)
    print("This will show only what the agent has explored.")
    print("Black areas = unexplored")
    print("White/colored areas = explored")
    print("\nPress Arrow keys to move, ESC to quit")
    print("=" * 60)
    
    import pygame
    running = True
    steps = 0
    
    # Get base env for rendering
    base_env = env.env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    while running and steps < 100:
        # Render with fog of war
        base_env.render(fog_of_war=True)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                action = None
                
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    
                    if terminated:
                        print(f"\n✓ Goal reached in {steps} steps!")
                        running = False
                        break
        
        pygame.time.wait(30)
    
    base_env.close()
    print("\nTest complete!")


if __name__ == "__main__":
    test_fog_of_war_rendering()
