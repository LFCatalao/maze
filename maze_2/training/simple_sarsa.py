import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class SimpleSARSA:
    def __init__(self, policy, env, learning_rate=1e-4, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.05, verbose=0, tensorboard_log=None, seed=None, **kwargs):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.num_timesteps = 0
        
        # Observation space is Box(12,)
        # Action space is Discrete(5)
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        self.q_net = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            q_values = self.q_net(obs_tensor)
            if deterministic:
                action = torch.argmax(q_values, dim=1).cpu().numpy()
            else:
                action = torch.argmax(q_values, dim=1).cpu().numpy() # Placeholder for epsilon-greedy if needed
        return action, None

    def learn(self, total_timesteps, callback=None, progress_bar=False):
        obs = self.env.reset()
        # DummyVecEnv returns numpy array
        
        # Initial action
        if np.random.rand() < 1.0: 
             action = [self.env.action_space.sample()]
        else:
             action, _ = self.predict(obs, deterministic=True)

        epsilon = 1.0
        
        if callback:
            callback.init_callback(self)
        
        for step in range(total_timesteps):
            self.num_timesteps += 1
            
            # Decay epsilon
            fraction = min(1.0, step / (total_timesteps * self.exploration_fraction))
            epsilon = 1.0 + fraction * (self.exploration_final_eps - 1.0)
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            
            # Choose next action (On-policy)
            if np.random.rand() < epsilon:
                next_action = [self.env.action_space.sample()]
            else:
                next_action, _ = self.predict(next_obs, deterministic=True)
            
            # SARSA Update
            real_next_obs = next_obs
            if done[0]:
                real_next_obs = info[0]['terminal_observation']
                real_next_obs = np.array([real_next_obs]) # Wrap in batch dim
            
            with torch.no_grad():
                next_obs_tensor = torch.FloatTensor(real_next_obs)
                next_q_values = self.q_net(next_obs_tensor)
                
                if done[0]:
                    target = reward[0]
                else:
                    next_q_val = next_q_values[0, next_action[0]]
                    target = reward[0] + self.gamma * next_q_val
            
            obs_tensor = torch.FloatTensor(obs)
            q_values = self.q_net(obs_tensor)
            current_q_val = q_values[0, action[0]]
            
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            
            loss = self.loss_fn(current_q_val, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            obs = next_obs
            action = next_action
            
            if done[0]:
                # Reset is handled by DummyVecEnv automatically!
                # But we need to pick a new action for the new episode start
                if np.random.rand() < epsilon:
                    action = [self.env.action_space.sample()]
                else:
                    action, _ = self.predict(obs, deterministic=True)
                    
            if callback:
                callback.on_step()
                
    def save(self, path):
        torch.save(self.q_net.state_dict(), path + ".pth")
        
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
