import sys
import os
import argparse
import carla

from common.loggers.comet_logger import CometLogger
from projects.bridge.config.logger_config import CometLoggerConfig

import gym
# from algorithms import PPO, SAC
from algorithms import BASAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from projects.bridge.algorithm.bridge_env import LavaBridge




def main(args):

    device = f'cuda:{args.gpu}'

    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = ["ent_sweep", "0.9kg"])
    logger = CometLogger(logger_conf)

    env = LavaBridge()

    dummy_eval_env = DummyVecEnv([lambda: env])

    eval_callback = EvalCallback(dummy_eval_env,
                                best_model_save_path= os.path.join(logger.log_dir, "policy", "models"),
                                log_path=os.path.join(logger.log_dir, "policy"),
                                eval_freq=1000,
                                deterministic=False, render=False,
                                n_eval_episodes = 10)


    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=os.path.join(logger.log_dir, "policy/models"),
                                            name_prefix='rl_model')

    model = BASAC("MlpPolicy", env, verbose=1, carla_logger = logger, device = device, ent_coef_min = 1e-4, ent_coef_max = 5, k = 5)
    # model = SAC("MlpPolicy", env, verbose=1, carla_logger = logger, device = device, ent_coef = args.ent)

    model.learn(total_timesteps=10000000, callback = checkpoint_callback)
    # model.learn(total_timesteps=10000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--ent', type=str)
    args = parser.parse_args()
    main(args)

