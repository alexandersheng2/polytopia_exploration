import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.game import PolytopiaEnv


class GymPolytopiaEnv(gym.Env):
    """
    Thin gymnasium.Env adapter around PolytopiaEnv so it can be trained
    with Stable-Baselines3. Delegates all game logic to PolytopiaEnv;
    only translates between the two step()/reset() API shapes.
    """

    def __init__(self):
        super().__init__()
        self.env = PolytopiaEnv()

        obs, _ = self.env.reset()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(PolytopiaEnv.NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = self.env.grid.all_explored()
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()
