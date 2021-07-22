# morel imports
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os

from gym.core import ObservationWrapper

# Add to path for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

# Logger
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

from projects.morel_mopo.config.fake_env_config import DefaultFakeEnvConfig
from fake_envs.fake_env import FakeEnv

from projects.morel_mopo.scripts.collect_data import DataCollector
from projects.morel_mopo.algorithm.dynamics_init import DynamicsEnsemble
# Gym
import gym
import gym.spaces
from gym.core import ObservationWrapper

# Pytorch
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# Stable baselines PPO
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
# Dynamics 
from projects.morel_mopo.config.dynamics_ensemble_config import MBPODynamicsEnsembleConfig, BaseDynamicsEnsembleConfig, MBPODynamicsModuleConfig, BaseDynamicsModuleConfig
# Datamodule
from data_modules import OfflineCarlaDataModule
import copy





class MBPO:
    def __init__(self, args,
                logger):

        print('------------------Initializing MBPO Parameters-----------------')
        # hyperparams
        self.experiment_name = args.exp_name
        self.n_epochs = args.n_epochs                # N
        self.n_models = args.n_models                # ensemble size
        self.n_timesteps = args.n_timesteps          # environment steps per epoch
        self.n_rollouts = args.n_rollouts            # number model rollouts per environment step
        self.n_grad_updates = args.n_grad_updates    # same as number of steps
        self.batch_size = args.batch_size            # Batch size
        self.buffer_size = args.buffer_size          # replay buffer
        self.gpu = args.gpu


        self.dynamics_epochs = 200 # TODO 
        self.rollout_len = 1
        self.num_initial_samples = 30000
        self.num_samples_per_timestep = 1000


        print('-----------------------Initializing Logging ------------------')

        self.logger = logger
        self.log_dir = "/home/scratch/vccheng/carla_test"
        self.device = "cuda:{}".format(self.gpu)
        self.collect_data_path = '/zfsauton/datasets/ArgoRL/vccheng/collect_data_mbpo' 


    ''' MBPO algorithm'''
    def train(self):
        # 1000 env steps per epoch 

        ''' save this data to path, datamodule loads it in (no need env buffer)'''
        ''' use old data to train dynamics model?'''
        ''' dataset size limit '''

        print('------------------------Initial Exploration-------------')

        data_collector = DataCollector()
        
        # initial exploration, collect data from other policy
        data_collector.collect_data(path=self.collect_data_path,
                                    policy=None,
                                    n_samples=self.num_initial_samples,\
                                    carla_gpu=self.gpu)
                    
        print('------------------------Creating CarlaDataModule-------------')
        
        class TempDataModuleConfig():
            def __init__(self, collect_data_paths):
                self.dataset_paths = collect_data_paths
                self.batch_size = 512
                self.frame_stack = 2
                self.num_workers = 2
                self.train_val_split = 0.95
                self.normalize_data = False

        # data config
        data_config = TempDataModuleConfig([self.collect_data_path])
        
        self.data_module = OfflineCarlaDataModule(data_config)

        # dynamics ensemble
        dyn_ensemble_config = MBPODynamicsEnsembleConfig()
        dyn_module_config = MBPODynamicsModuleConfig()

        print('--------------------Initializing Dynamics Ensemble------------')
        self.dynamics = DynamicsEnsemble(
            config=dyn_ensemble_config,
            data_module = self.data_module,
            state_dim_in = dyn_module_config.state_dim_in,
            state_dim_out = dyn_module_config.state_dim_out,
            action_dim = 2,
            frame_stack = dyn_module_config.frame_stack,
            norm_stats = self.data_module.normalization_stats,
            gpu = self.gpu,
            logger = self.logger,
            log_freq = 100)
       

        print('---------------------Initializing FakeEnv ----------------')

        fake_env_config = DefaultFakeEnvConfig()
        fake_env_config.populate_config(\
            observation_config = "DefaultObservationConfig", \
            action_config = "DefaultActionConfig",\
            reward_config="DefaultRewardConfig",\
            uncertainty_config="DefaultUncertaintyConfig")
            
        self.fake_env = FakeEnv(self.dynamics,
                        config = fake_env_config,
                        logger = self.logger)
        
        print('--------------------------Initializing Policy-----------------')

        self.policy = SAC(policy='MlpPolicy', 
                          env = self.fake_env,
                          batch_size = self.batch_size,\
                          buffer_size = self.buffer_size,
                          learning_starts = 0,\

                          # as soon as collected_steps >= k, stop
                          train_freq=(self.rollout_len, "step"),\
                          carla_logger = self.logger,\
                          gradient_steps = 1,\
                          verbose = 1)

        print('---------------------Beginning MBPO Training------------------')


        for epoch in range(self.n_epochs):
            print(f'Epoch {epoch}')

            # train a predictive model on D_env via max likelihood 
            self.dynamics.train(self.dynamics_epochs)

            for timestep in range(self.n_timesteps): # E

                # take actions in real env, update D_env
                new_path = f"{self.collect_data_path}_{epoch}"
                data_collector.collect_data(path= new_path,
                                            policy=self.policy,
                                            n_samples=self.num_samples_per_timestep,
                                            carla_gpu=self.gpu)
                # load in newly collected data (DynamicsEnsemble, FakeEnv should update automatically)
                self.data_module.update(new_path)


                policy_timesteps = self.n_rollouts * self.rollout_len 
                print(f"------Begin policy rollouts for {policy_timesteps} timesteps")
                self.policy.learn(total_timesteps = policy_timesteps)


def main(args):
    # import pdb; pdb.set_trace()
    EXPERIMENT_NAME = args.exp_name
    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["MBPO"])
    logger = CometLogger(logger_conf)
    print('logger.logdir', logger.log_dir)
    m = MBPO(args, logger=logger)
    m.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # num epochs
    parser.add_argument('--n_epochs', type=int, default=400)
    # ensemble size
    parser.add_argument('--n_models', type=int, default=7)
    # environment steps per epoch
    parser.add_argument('--n_timesteps', type=int, default=1000)
    # model rollouts per environment step
    parser.add_argument('--n_rollouts', type=int, default=400)
    # gradient updates
    parser.add_argument('--n_grad_updates', type=int, default=40)
    # batch
    parser.add_argument('--batch_size', type=int, default=512)
    # replay buffer size
    parser.add_argument('--buffer_size', type=int, default=1000000)
    # load
    parser.add_argument('--load', action='store_true', default=False)
    # gpu 
    parser.add_argument('--gpu', type=int, default=2)
    # experiment name
    parser.add_argument('--exp_name', default="MBPO")

    args = parser.parse_args()
    main(args)




