import sys
import os

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

import gym
from algorithms import PPO
from stable_baselines3.common.env_util import DummyVecEnv

# Environment
from environment.env import CarlaEnv
from environment.config import ConfigManager


config_manager = ConfigManager(algo="PPO")
env = CarlaEnv(config = config_manager.config, log_dir = "/home/scratch/swapnilp/carla_test")
# Parallel environments
dummy_env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

