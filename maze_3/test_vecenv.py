from environments import LockedRoomEnv, apply_reward_wrapper, SimpleObs
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env():
    env = LockedRoomEnv(size=19, num_doors=0, fixed_agent_pos=(9, 9), fixed_goal_pos=(3, 3))
    env = apply_reward_wrapper(env, 'simple')
    env = SimpleObs(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

obs = env.reset()
print(f'VecEnv obs type: {type(obs)}')
print(f'VecEnv obs shape: {obs.shape}')
print(f'VecEnv obs dtype: {obs.dtype}')
print(f'VecEnv obs: {obs}')

# Try a step
action = env.action_space.sample()
obs, reward, done, info = env.step([action])
print(f'\nAfter step:')
print(f'Obs type: {type(obs)}')
print(f'Obs shape: {obs.shape}')
print(f'Obs dtype: {obs.dtype}')
