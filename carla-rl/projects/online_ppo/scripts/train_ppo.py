import sys
import os

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


def main(args):
    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = ["Online_PPO"])


    logger = CometLogger(logger_conf)
    print(logger.log_dir)

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashRegularTown01Config",
        testing = False,
        carla_gpu = args.gpu
    )
    # logger_callback = PPOLoggerCallback(logger)


    env = CarlaEnv(config = config, logger = logger, log_dir = "/home/scratch/swapnilp/carla-rl_testing")

    eval_env = env.get_eval_env(eval_frequency = 5000)
    dummy_eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(dummy_eval_env, best_model_save_path=os.path.join(logger.log_dir, "policy", "models"),
                                log_path=os.path.join(logger.log_dir, "policy"), eval_freq=5000,
                                deterministic=False, render=False,
                                n_eval_episodes=config.scenario_config.num_episodes)



    tb_log_dir = os.path.join(logger.log_dir, "ppo_tb_logs")
    os.makedirs(tb_log_dir)

    import ipdb; ipdb.set_trace()
    model = PPO("MlpPolicy", env, verbose=1, carla_logger = logger)
    model.learn(total_timesteps=10000000, callback = eval_callback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    main(args)


