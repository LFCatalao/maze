# Save as debug_reward4.py
from environments import LockedRoomEnv, get_config, apply_reward_wrapper
from training.train_experiment import FlattenObservation

config = get_config("E3")

env = LockedRoomEnv(
    size=config.size,
    observation_mode="full_map",
    render_mode=None,
    max_steps=400,
    fixed_agent_pos=config.fixed_agent_pos,
    fixed_goal_pos=config.fixed_goal_pos,
    num_doors=config.num_doors,
    include_key=config.include_key,
    locked_door=config.locked_door,
)
env = apply_reward_wrapper(env, "simple")
env = FlattenObservation(env)

obs, info = env.reset()
base_env = env.env.env

key_pos = list(base_env.key_positions.keys())[0]
agent_pos = base_env.agent_pos

print(f"Agent: {agent_pos}")
print(f"Key: {key_pos}")
print(f"Goal: {base_env.goal_pos}")
print()

# Check what distance_to_target returns
reward_wrapper = env.env  # The SimpleRewardWrapper
print(f"Current target distance: {reward_wrapper._distance_to_target()}")
print(f"Current target (should be adjacent to key): calculated from key at {key_pos}")
print(
    f"Adjacent positions: above=({key_pos[0]-1}, {key_pos[1]}), below=({key_pos[0]+1}, {key_pos[1]}), left=({key_pos[0]}, {key_pos[1]-1}), right=({key_pos[0]}, {key_pos[1]+1})"
)
print()

# Test a few random actions and see rewards
import random

action_names = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "TOGGLE"]

print("Testing 20 random steps:")
for i in range(20):
    action = random.randint(0, 2)  # Only movement actions
    old_pos = list(base_env.agent_pos)
    old_dist = reward_wrapper._distance_to_target()

    obs, reward, term, trunc, info = env.step(action)

    new_pos = list(base_env.agent_pos)
    new_dist = reward_wrapper._distance_to_target()

    print(
        f"  {action_names[action]:8} | {old_pos} -> {new_pos} | dist: {old_dist} -> {new_dist} | reward: {reward:+.2f}"
    )

    if term:
        break

env.close()
