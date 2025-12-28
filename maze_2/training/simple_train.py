from environments import LockedRoomEnv, get_config, apply_reward_wrapper
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 20, (5,), np.float32)

    def observation(self, obs):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        return np.array(
            [
                base_env.agent_pos[0],
                base_env.agent_pos[1],
                base_env.goal_pos[0],
                base_env.goal_pos[1],
                base_env.agent_dir,
            ],
            dtype=np.float32,
        )


def make_env():
    config = get_config("E1")
    env = LockedRoomEnv(
        size=config.size,
        observation_mode="full_map",
        render_mode=None,
        max_steps=200,
        fixed_agent_pos=config.fixed_agent_pos,
        fixed_goal_pos=config.fixed_goal_pos,
        num_doors=0,
    )
    env = apply_reward_wrapper(env, "simple")
    return SimpleObs(env)


# Train
env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)
model.learn(total_timesteps=50000)
model.save("simple_obs_model")

# Evaluate
eval_env = make_env()
successes = 0
for _ in range(100):
    obs, _ = eval_env.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, _, _ = eval_env.step(action)
        if term:
            successes += 1
            break

print(f"Success rate: {successes}%")
