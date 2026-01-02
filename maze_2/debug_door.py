# Save as debug_door.py
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
reward_wrapper = env.env

# Give agent the key and position near door
base_env.carrying = 4  # Yellow key
base_env.agent_pos = [3, 9]  # In corridor, near door at (3, 7)
base_env.agent_dir = 2  # Facing LEFT toward door

reward_wrapper.previous_distance = reward_wrapper._distance_to_target()
reward_wrapper.gave_pickup_hint = True  # Already got key

print(f"Agent at {base_env.agent_pos}, facing LEFT")
print(f"Door at (3, 7)")
print(f"Phase: {reward_wrapper._get_phase()}")
print(f"Target: {reward_wrapper._get_current_target()}")
print(f"Distance to target: {reward_wrapper._distance_to_target()}")
print()

# Test moving toward door
action_names = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "TOGGLE"]

print("Testing steps toward door:")
for i in range(10):
    action = 2  # FORWARD
    old_pos = list(base_env.agent_pos)
    old_dist = reward_wrapper._distance_to_target()

    obs, reward, term, trunc, info = env.step(action)

    new_pos = list(base_env.agent_pos)
    new_dist = reward_wrapper._distance_to_target()
    can_open = obs[11]

    print(
        f"  {action_names[action]:8} | {old_pos} -> {new_pos} | dist: {old_dist}->{new_dist} | reward: {reward:+.2f} | can_open: {can_open}"
    )

    if can_open == 1.0:
        print(f"\n  *** CAN OPEN DOOR! Pressing TOGGLE ***")
        obs, reward, term, trunc, info = env.step(4)  # TOGGLE
        print(f"  TOGGLE reward: {reward:+.2f}")
        print(f"  Door state: {base_env.door_positions}")
        break

env.close()
