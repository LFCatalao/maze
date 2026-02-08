from environments import LockedRoomEnv, apply_reward_wrapper, SimpleObs

env = LockedRoomEnv(size=19, num_doors=0)
env = apply_reward_wrapper(env, 'simple')
env = SimpleObs(env)

obs, info = env.reset()
print(f'Obs type: {type(obs)}')
print(f'Obs dtype: {obs.dtype}')
print(f'Obs shape: {obs.shape}')
print(f'Obs: {obs}')
