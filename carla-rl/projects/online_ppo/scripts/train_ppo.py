import sys
import os
import argparse

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.online_ppo.config.logger_config import CometLoggerConfig

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig


def main(args):
    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = ["Online_PPO"])


    # logger = CometLogger(logger_conf)
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
    # logger_callback = PPOLoggerCallback(logger)


    env = CarlaEnv(config = config, logger = None, log_dir = "test")

    while True:
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

    # eval_env = env.get_eval_env(eval_frequency = 5000)
    # dummy_eval_env = DummyVecEnv([lambda: eval_env])


    # best_model_save_path = os.path.join(logger.log_dir, "policy", "models")
    # eval_callback = EvalCallback(dummy_eval_env,
    #                             best_model_save_path=best_model_save_path,
    #                             log_path=os.path.join(logger.log_dir, "policy"),
    #                             eval_freq=1000,
    #                             deterministic=False, render=False,
    #                             n_eval_episodes=config.scenario_config.num_episodes)



    # tb_log_dir = os.path.join(logger.log_dir, "ppo_tb_logs")
    # os.makedirs(tb_log_dir)


    model = PPO("MlpPolicy", env, verbose=1, carla_logger = None)

    # if args.load:
        # print('Loading from ', best_model_save_path)
        # model = PPO.load(best_model_save_path)
    # model.learn(total_timesteps=10000000, callback = eval_callback)
    model.learn(total_timesteps=10000000)


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

