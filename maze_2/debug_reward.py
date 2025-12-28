from environments import LockedRoomEnv, get_config, apply_reward_wrapper
from training.train_experiment import FlattenObservation

config = get_config("E3")


def make_env():
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
    return FlattenObservation(env)


env = make_env()
obs, info = env.reset()

print("Observation breakdown:")
print(f"  agent_y, agent_x: {obs[0]}, {obs[1]}")
print(f"  goal_y, goal_x: {obs[2]}, {obs[3]}")
print(f"  direction: {obs[4]}")
print(f"  has_key: {obs[5]}")
print(f"  key_y, key_x: {obs[6]}, {obs[7]}")
print(f"  door_y, door_x: {obs[8]}, {obs[9]}")
if len(obs) > 10:
    print(f"  on_key: {obs[10]}")
    print(f"  on_door: {obs[11]}")

# Get base environment
base_env = env.env.env

print(f"\nKey position: {list(base_env.key_positions.keys())[0]}")
print(f"Goal position: {base_env.goal_pos}")
print(f"Agent position: {base_env.agent_pos}")
print(f"Carrying: {base_env.carrying}")

# Simulate steps
action_names = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "TOGGLE"]
print("\nSimulating 10 FORWARD steps:")
for i in range(10):
    action = 2  # FORWARD
    old_pos = list(base_env.agent_pos)
    obs, reward, term, trunc, info = env.step(action)
    new_pos = list(base_env.agent_pos)
    has_key = base_env.carrying is not None
    print(f"  Step {i+1}: {old_pos} -> {new_pos} | reward: {reward:+.2f} | has_key: {has_key}")

env.close()
