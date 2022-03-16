import sys
import os
import argparse

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

import carla
import random

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.algorithm.mopo_parallel import MOPO
from projects.morel_mopo.algorithm.dist_utils import run_param_server, get_host_ip
from projects.morel_mopo.config.morel_mopo_config import DefaultMLPMOPOConfig, DefaultProbMLPMOPOConfig, DefaultProbGRUMOPOConfig, DefaultMLPObstaclesMOPOConfig


def launch_server(rank, resources):
    os.environ['RANK'] = str(rank)
    model = MOPO(config=resources['config'],
                logger=resources['logger'])
    # train MOReL
    model.serve()

def launch_worker(rank, resources):
    os.environ['RANK'] = str(rank)
    model = MOPO(config=resources['config'],
                logger=resources['logger'])
    # train MOReL
    model.work()


def main(args):
    ########################################## logger  ####################################

    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = ["MOPO", "uncertainty_sweep"])
    logger = CometLogger(logger_conf)

    # Load the dynamics from a pre-verified dynamics model
    config = DefaultMLPObstaclesMOPOConfig()
    config.populate_config(gpu = args.gpu,
                           policy_algorithm = "PPO",
                           pretrained_dynamics_model_key = "e1a27faf07f9450a87e6e6c10f29b0d8",
                           pretrained_dynamics_model_name = "final")

    # config.populate_config(gpu = args.gpu, policy_algorithm = "SAC")

    config.fake_env_config.uncertainty_coeff = args.uncertainty
    config.fake_env_config.uncertainty_coeff = 1

    run_param_server(launch_server, launch_worker, 1, 3,
        {'config': config, 'logger': logger}, get_host_ip(),
        random.randint(10000, 60000), mp_method='fork')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--uncertainty", type=float)
    args = parser.parse_args()
    main(args)



