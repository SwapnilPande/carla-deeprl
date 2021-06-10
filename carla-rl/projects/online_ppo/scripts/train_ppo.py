import sys
import os

from gym.core import ObservationWrapper

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

import gym
from algorithms import PPO
from stable_baselines3.common.env_util import DummyVecEnv

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig


config = DefaultMainConfig()
config.populate_config(
    observation_config = "LowDimObservationConfig",
    action_config = "MergedSpeedScaledTanhConfig",
    reward_config = "Simple2RewardConfig",
    scenario_config = "NoCrashEmptyTown01Config",
    testing = False,
    carla_gpu = 0
)


env = CarlaEnv(config = config, log_dir = "/home/scratch/swapnilp/carla_test")
# Parallel environments
dummy_env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

