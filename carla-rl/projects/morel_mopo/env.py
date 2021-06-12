import gym
import numpy as np
import torch
class GymWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    super(GymWrapper, self).__init__(env)
    self.env = env

  
  def reset(self, obs, action):
    """
    Resets the environment 
    """
    self.env.reset(obs, action)

  def step(action):
    obs = self.env.step(action)
    return obs
