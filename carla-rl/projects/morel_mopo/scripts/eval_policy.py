import carla

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

from stable_baselines3 import PPO, SAC




config = DefaultMainConfig()
config.populate_config(
    observation_config = "VehicleDynamicsNoCameraConfig",
    action_config = "MergedSpeedTanhConfig",
    reward_config="Simple2RewardConfig",
    scenario_config = "NoCrashEmptyTown01Config",
    testing = False,
    carla_gpu = 1
)
# logger_callback = PPOLoggerCallback(logger)


env = CarlaEnv(config = config, logger = None, log_dir = "/home/swapnil/carla_logs")


policy = PPO.load("/home/swapnil/best_model.zip")

while True:
    obs = env.reset()
    for i in range(1000):
        action, _states = policy.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
