import sys
import os
import argparse

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

import carla

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.algorithm.mopo import MOPO
from projects.morel_mopo.config.morel_mopo_config import DefaultMLPMOPOConfig



def main(args):
    ########################################## logger  ####################################

    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = args.exp_name, tags = ["MOPO"])
    logger = CometLogger(logger_conf)

    config = DefaultMLPMOPOConfig()
    config.populate_config(gpu = args.gpu)

    model = MOPO(config = config,
                logger=logger)

    # train MOReL
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    main(args)


