import sys
import os
import argparse
from typing import Optional

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

import carla

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.algorithm.mopo import MOPO
from projects.morel_mopo.config.morel_mopo_config import DefaultMLPMOPOConfig, DefaultProbMLPMOPOConfig, DefaultProbGRUMOPOConfig, DefaultMLPObstaclesMOPOConfig



def main(args):
    ########################################## logger  ####################################

    logger_conf = CometLoggerConfig()
    # logger_conf.disable = True
    logger_conf.populate(experiment_name = args.exp_name)
    logger = CometLogger(logger_conf)

    # Check if exp_group in args
    if args.exp_group is not None:
        # Log exp_group as a hyperparameter
        logger.log_hyperparameters({'exp_group', args.exp_group})

    # Load the dynamics from a pre-verified dynamics model
    config = DefaultMLPObstaclesMOPOConfig()
    config.populate_config(gpu = args.gpu,
                           policy_algorithm = "PPO",
                           pretrained_dynamics_model_key = "e1a27faf07f9450a87e6e6c10f29b0d8",
                           pretrained_dynamics_model_name = "final")

    # config.populate_config(gpu = args.gpu, policy_algorithm = "SAC")

    config.fake_env_config.uncertainty_coeff = args.uncertainty
    config.fake_env_config.uncertainty_coeff = 1

    model = MOPO(config = config,
                logger=logger)

    # train MOReL
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--uncertainty", type=float)
    parser.add_argument("--exp_group", type=str, required = False)
    # Parse known args
    args = parser.parse_known_args()[0]
    main(args)


