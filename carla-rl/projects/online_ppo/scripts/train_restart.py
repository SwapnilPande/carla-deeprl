import sys
import os
import argparse

import carla

import faulthandler
faulthandler.enable()

# import ipdb; ipdb.set_trace()

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.tensorboard_logger import TensorboardLogger
from projects.online_ppo.config.logger_config import TensorboardLoggerConfig, ExistingTensorboardLoggerConfig

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import DummyVecEnv
from algorithms import train_restart_wrapper

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import time

def main(args):
    logger_conf = TensorboardLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name)
    logger =  TensorboardLogger(logger_conf)
    exp_key = logger.experiment_key

    def train_fn(train_steps_per_restart, model_save_name, model_load_name = None):
        device = f"cuda:{args.gpu}"
        print(f"Using device: {device}")

        # Configure logger
        logger_conf = ExistingTensorboardLoggerConfig()
        logger_conf.experiment_key = exp_key
        logger = TensorboardLogger(logger_conf)

        print(logger.log_dir)

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

        env = CarlaEnv(config = config, logger = None, log_dir = os.path.join(logger.log_dir, "env"))


        # TODO Eval callback is not included because switching between eval and train is not trivial for leaderboard
        # eval_env = env.get_eval_env(eval_frequency = 100000)
        # dummy_eval_env = DummyVecEnv([lambda: eval_env])

        # eval_callback = EvalCallback(dummy_eval_env, best_model_save_path=os.path.join(logger.log_dir, "policy", "models"),
        #                             log_path=os.path.join(logger.log_dir, "policy"), eval_freq=100000,
        #                             deterministic=False, render=False,
        #                             n_eval_episodes=config.scenario_config.num_episodes)

        if(model_load_name is not None):
            model_path = os.path.join(logger.log_dir, "policy", "models", model_load_name)
            print(f"LOADING MODEL FROM: {model_path}")
            model = PPO.load(model_path,
                            env = env,
                            device = device,
                            carla_logger=None)
        else:
            model = PPO("MlpPolicy", env, verbose=1, carla_logger = logger, device = device)
            # print(f"LOADING MODEL FROM: /home/scratch/swapnilp/leaderboard_policies_auto_restart/policy/models/policy_3000000_steps.zip")
            # model = PPO.load("/home/scratch/swapnilp/leaderboard_policies_auto_restart/policy/models/policy_3000000_steps.zip",
            #                 env = env,
            #                 device = device,
            #                 carla_logger=None,
            #                 ent_coef = 0.01)

        # Try to train, close environment on crash
        try:
            model.learn(total_timesteps = train_steps_per_restart)
        except Exception as e:
            print("Environment has crashed... Restarting training from the previous checkpoint!")
            env.close()
            time.sleep(5)
            raise e


        model.save(os.path.join(logger.log_dir, "policy", "models", model_save_name))

        env.close()
        time.sleep(5)

        return True

    train_restart_wrapper(train_fn, 10000000, 100000)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    main(args)

