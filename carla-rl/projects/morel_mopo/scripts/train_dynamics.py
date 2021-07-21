import sys
import os
import argparse

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from projects.morel_mopo.config.logger_config import CometLoggerConfig


# Import DataModuleConfig


from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.dynamics_config import DefaultMLPDynamicsConfig, DefaultDeterministicGRUDynamicsConfig, DefaultProbabilisticGRUDynamicsConfig
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule, RNNOfflineCarlaDataModule
# from projects.morel_mopo.algorithm.fake_env import FakeEnv
# from projects.morel_mopo.config.fake_env_config import DefaultMainConfig

import torch
TAGS = ["dyn_only", "GRU"]

def main(args):

    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = TAGS)


    logger = CometLogger(logger_conf)


    # Create config for MLP dynamics
    # dynamics_config = DefaultMLPDynamicsConfig()
    dynamics_config = DefaultProbabilisticGRUDynamicsConfig()
    dynamics_config.populate_config(gpu = args.gpu)

    ######### MLP
    # data config
    data_module = dynamics_config.dataset_type(dynamics_config.dataset_config)

    # dynamics config
    # dyn_ensemble_config = DefaultDynamicsEnsembleConfig()
    # dyn_ensemble_config.gpu = args.gpu

    dynamics_config.dynamics_model_config.gpu = args.gpu

    # dynamics = ProbabilisticDynamicsGRUEnsemble(
    #         config = dyn_ensemble_config,
    #         data_module = data_module,
    #         logger = logger
    #     )

    dynamics = dynamics_config.dynamics_model_type(
            config = dynamics_config.dynamics_model_config,
            data_module = data_module,
            logger = logger
        )

    # Train for 500 epochs
    dynamics.train_model(dynamics_config.train_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    main(args)

