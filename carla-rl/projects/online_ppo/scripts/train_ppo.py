import sys
import os
import argparse
import carla

import faulthandler
faulthandler.enable()

# import ipdb; ipdb.set_trace()

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.online_ppo.config.logger_config import CometLoggerConfig

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import time

def main(args):
    # logger_conf = CometLoggerConfig()
    # logger_conf.populate(experiment_name = args.exp_name, tags = ["leaderboard"])

    device = f"cuda:{args.gpu}"


    # logger =  CometLogger(logger_conf)
    # print(logger.log_dir)

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationNoCameraConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "LeaderboardConfig",
        testing = False,
        carla_gpu = args.gpu
    )
    # config.verify()
    # logger_callback = PPOLoggerCallback(logger)

    env = CarlaEnv(config = config, logger = None, log_dir = "/home/scratch/swapnilp/leaderboard_policies_actors_2")

    # eval_env = env.get_eval_env(eval_frequency = 100000)
    # dummy_eval_env = DummyVecEnv([lambda: eval_env])

    # eval_callback = EvalCallback(dummy_eval_env, best_model_save_path=os.path.join(logger.log_dir, "policy", "models"),
    #                             log_path=os.path.join(logger.log_dir, "policy"), eval_freq=100000,
    #                             deterministic=False, render=False,
    #                             n_eval_episodes=config.scenario_config.num_episodes)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000000//50, save_path=os.path.join("/home/scratch/swapnilp/leaderboard_policies_actors_2", "policy", "models"),
                                                name_prefix='policy_checkpoint_')

    model_path = "/home/scratch/swapnilp/next_policy_400000_steps.zip"
    print(f"LOADING MODEL FROM: {model_path}")
    model = PPO.load(model_path,
                    env = env,
                    device = device,
                    carla_logger=None,
                    ent_coef = 0.01)#PPO("MlpPolicy", env, verbose=1, carla_logger = None, device = device)


    callbacks = [checkpoint_callback]
    model.learn(total_timesteps=10000000, callback = callbacks)

    model.save(os.path.join("/home/scratch/swapnilp/leaderboard_policies_actors_2", "policy", "models", "final_policy"))

# def save(self, save_dir):
#     if(not os.path.isdir(save_dir)):
#         os.mkdir(save_dir)

#     self.policy.save(save_dir)
#     self.dynamics.save(save_dir)



# def load(self, load_dir):
#     self.policy.load(load_dir)
#     # self.dynamics.load(load_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()
    main(args)

