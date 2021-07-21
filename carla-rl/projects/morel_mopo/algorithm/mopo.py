# morel imports
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../../')))

from environment.env import CarlaEnv


# Stable baselines PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3 import PPO

# Environment
from stable_baselines3.common.callbacks import EvalCallback



'''
1. Learn approx dynamics model from OfflineCarlaDataset
3. Construct pessimistic MDP FakeEnv with USAD detector
4. Train policy in FakeEnv
'''
class MOPO():
    def __init__(self, config,
                    logger):
        self.config = config
        self.dynamics_config = self.config.dynamics_config
        self.fake_env_config = self.config.fake_env_config
        self.eval_env_config = self.config.eval_env_config

        self.dynamics_epochs = self.config.dynamics_config.train_epochs
        self.policy_epochs = 10_000_000

        self.logger = logger

        self.data_module = self.dynamics_config.dataset_type(self.dynamics_config.dataset_config)

        self.dynamics = self.dynamics_config.dynamics_model_type(
            config = self.dynamics_config.dynamics_model_config,
            data_module = self.data_module,
            logger = logger
        )

        self.fake_env = None

        self.policy  = None

    def train(self):

        # Log MOPO hyperparameters
        self.logger.log_hyperparameters({
            "mopo/uncertainty_penalty" : self.fake_env_config.uncertainty_coeff,
            "mopo/rollout_length" : self.fake_env_config.timeout_steps,
        })


        print("MOPO: Beginning Dynamics Training")

        self.dynamics.train_model(self.dynamics_epochs)

        print("MOPO: Constructing Fake Env")


        fake_env = self.dynamics_config.fake_env_type(self.dynamics,
                        config = self.fake_env_config,
                        logger = self.logger)

        print("MOPO: Constructing Real Env for evaluation")


        env = CarlaEnv(config = self.eval_env_config, logger = self.logger, log_dir = self.logger.log_dir)
        eval_env = env.get_eval_env(eval_frequency = 5000)
        dummy_eval_env = DummyVecEnv([lambda: eval_env])

        eval_callback = EvalCallback(dummy_eval_env, best_model_save_path=os.path.join(self.logger.log_dir, "policy", "models"),
                                    log_path=os.path.join(self.logger.log_dir, "policy"), eval_freq=5000,
                                    deterministic=False, render=False,
                                    n_eval_episodes=self.eval_env_config.scenario_config.num_episodes)

        print("MOPO: Beginning Policy Training")

        import ipdb; ipdb.set_trace()
        self.policy = PPO("MlpPolicy", fake_env, verbose=1, carla_logger = self.logger)
        self.policy.learn(total_timesteps=self.policy_epochs, callback = eval_callback)


    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)




    def load(self, load_dir):
        self.policy.load(load_dir)
        # self.dynamics.load(load_dir)
