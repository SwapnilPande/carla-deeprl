import sys
import os
import argparse

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from projects.morel_mopo.config.logger_config import CometLoggerConfig
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, DefaultGRUDynamicsConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
from projects.morel_mopo.algorithm.dynamics_gru import DynamicsGRUEnsemble
from projects.morel_mopo.algorithm.probabilistic_dynamics_gru import ProbabilisticDynamicsGRUEnsemble
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
# from projects.morel_mopo.algorithm.fake_env import FakeEnv
# from projects.morel_mopo.config.fake_env_config import DefaultMainConfig

import torch
TAGS = ["dyn_only", "GRU"]

def main(args):
    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = TAGS)


    logger = CometLogger(logger_conf)
    class TempDataModuleConfig():
        def __init__(self):
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random"
            ]

            self.batch_size = 512
            self.frame_stack = 5
            self.num_workers = 10
            self.train_val_split = 0.95
            # Whether or not to stack the output
            # Use if the dynamics model is an RNN
            # False if no stack (MLP), True if stack (RNN)
            self.stack_output = True

    # data config
    data_config = TempDataModuleConfig()
    data_module = OfflineCarlaDataModule(data_config)

    # dynamics config
    # dyn_ensemble_config = DefaultDynamicsEnsembleConfig()
    # dyn_ensemble_config.gpu = args.gpu

    dyn_ensemble_config = DefaultGRUDynamicsConfig()
    dyn_ensemble_config.gpu = args.gpu

    # dynamics = ProbabilisticDynamicsGRUEnsemble(
    #         config = dyn_ensemble_config,
    #         data_module = data_module,
    #         logger = logger
    #     )

    dynamics = DynamicsGRUEnsemble(
            config = dyn_ensemble_config,
            data_module = data_module,
            logger = logger
        )


    # Train for 500 epochs
    dynamics.train(200)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    main(args)

