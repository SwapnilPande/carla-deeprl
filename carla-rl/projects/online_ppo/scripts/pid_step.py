import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym.core import ObservationWrapper
# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import torch 

EXPERIMENT_NAME = "pid_kp0.35_kd0.0004_ki0.1"

logger_conf = CometLoggerConfig()
logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["PID"])


logger = CometLogger(logger_conf)
print(logger.log_dir)

config = DefaultMainConfig()
config.populate_config(
    observation_config = "LowDimObservationConfig",
    action_config = "MergedSpeedScaledTanhConfig",
    reward_config = "Simple2RewardConfig",
    scenario_config = "NoCrashEmptyTown01Config",
    testing = False,
    carla_gpu = 2
)
# logger_callback = PPOLoggerCallback(logger)


env = CarlaEnv(config = config, logger = logger, log_dir = "/home/scratch/vccheng/carla_test")

# pid = env.carla_interface.actor_fleet.args_longitudinal_dict
# print(pid)

i = 0

done= False
while not done:
    target_speeds, speeds, control_throttles, control_brakes = [], [], [], []
    steps = 0
    env.reset()
    done = False 
    while not done:
        i += 1
        new_obs, rewards, done, infos = env.step(torch.Tensor([[0, 0.5]]))

        logger.log_scalar('target_speed', infos['target_speed'], steps)
        logger.log_scalar("speed", infos['speed'], steps)
        logger.log_scalar('control_throttle', infos['control_throttle'], steps)
        logger.log_scalar('control_brake', infos['control_brake'], steps)

        target_speeds.append(infos['target_speed'])
        speeds.append(infos['speed'])

        control_throttles.append(infos["control_throttle"])
        control_brakes.append(infos["control_brake"])


        if i > 10000 and i % 10000 == 0:
            print('target speed', target_speed)
            print('speed', speed)
        
        steps += 1 
        if done:
            print(f"done episode after {steps} steps")
            plt.plot(range(steps), target_speeds)
            plt.plot(range(steps), speeds)
            plt.show()


    # target_speeds.append(infos['target_speed'])
    # speeds.append(infos['speed'])

    # control_throttles.append(infos["control_throttle"])
    # control_brakes.append(infos["control_brake"])

    steps += 1
