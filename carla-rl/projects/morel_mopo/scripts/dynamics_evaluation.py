import sys
import os

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule

EXPERIMENT_NAME = "first_test"
TAGS = ["dyn_only"]




def n_step_eval(real_env, fake_env, policy, n =1):
    raise NotImplementedError


def main():
    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = "c2b22966acdf49faa5e2578cd925e653"

    logger = CometLogger(logger_conf)

    config = logger.pickle_load("dynamics_ensemble", "config.pkl")

    import ipdb; ipdb.set_trace()

    # dyn_conf = DefaultDynamicsEnsembleConfig()


    # if train:
    #     train_dynamics(data_module_conf = data_conf,
    #                     logger_conf = logger_conf,
    #                     dynamics_conf = dyn_conf)


    # Retrieve experiment

    # Set up data module

    # Retrieve latest model

    # Run desired experiments





if __name__ == "__main__":
    main()

