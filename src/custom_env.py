import os
import pandas as pd
import gymnasium
from gymnasium import spaces
import numpy as np

class CustomTradingEnv(gymnasium.Env):
    def __init__(self, data):
        super(CustomTradingEnv, self).__init__()
        self.data = data
        self.prepare_data()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.data.columns),), dtype=np.float32)
        self.current_step = 0

    def prepare_data(self):

        if 'Datetime' in self.data.columns:
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
            self.data['Datetime'] = self.data['Datetime'].apply(lambda x: x.timestamp())
        
        # Convert all data to numeric, coerce errors and fill NaNs with a predefined value like 0.0
        self.data = self.data.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        obs = np.nan_to_num(obs)  
        info = {} 
        return obs, info  

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        obs = self.data.iloc[self.current_step].values.astype(np.float32) if not terminated else None
        if obs is not None:
            obs = np.nan_to_num(obs) 
        reward = self.calculate_reward(action)
        info = {} 
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def calculate_reward(self, action):
        reward = np.random.randn() 
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0 
        return reward

    def close(self):
        pass