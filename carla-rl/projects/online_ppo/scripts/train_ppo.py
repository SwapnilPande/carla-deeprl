import sys
import os
import argparse
import carla

import faulthandler
faulthandler.enable()

from common.loggers.comet_logger import CometLogger
from projects.online_ppo.config.logger_config import CometLoggerConfig

import gym
from algorithms import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import time

def main(args):
    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = ["no_crash_dense"])

    device = f"cuda:{args.gpu}"


    logger =  CometLogger(logger_conf)
    print(logger.log_dir)

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationNoCameraConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashRegularTown01Config",
        testing = False,
        carla_gpu = args.gpu
    )
    # config.verify()
    # logger_callback = PPOLoggerCallback(logger)

    env = CarlaEnv(config = config, logger = logger)

    eval_env = env.get_eval_env(eval_frequency = 200000)
    dummy_eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(dummy_eval_env, best_model_save_path=os.path.join(logger.log_dir, "policy", "models"),
                                log_path=os.path.join(logger.log_dir, "policy"), eval_freq=200000,
                                deterministic=True, render=False,
                                n_eval_episodes=config.scenario_config.num_episodes)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=10000000//50, save_path=os.path.join(logger.log_dir, "policy", "models"),
                                                name_prefix='policy_checkpoint_')

    if(args.load):
        print(f"LOADING MODEL FROM: {args.load}")
        model = PPO.load(model_path,
                    env = env,
                    device = device,
                    carla_logger = logger)
    else:
        model = PPO("MlpPolicy", env, verbose=1, carla_logger = logger, device = device)


    callbacks = [checkpoint_callback, eval_callback]
    model.learn(total_timesteps=10000000, callback = callbacks)

    model.save(os.path.join(logger.log_dir, "policy", "models", "final_policy"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--load', type=str, default = None)
    args = parser.parse_args()
    main(args)

