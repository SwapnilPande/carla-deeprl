import sys
import os
# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, BaseDynamicsEnsembleConfig, DefaultDynamicsModuleConfig, BaseDynamicsModuleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
from projects.morel_mopo.algorithm.fake_env import FakeEnv
from projects.morel_mopo.config.fake_env_config import DefaultMainConfig

import torch
EXPERIMENT_NAME = "Vivian - Dynamics eval with uncertainty penalty"
TAGS = ["dyn_only"]

def main():
    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = TAGS)


    logger = CometLogger(logger_conf)
    class TempDataModuleConfig():
        def __init__(self):
            self.dataset_paths = ["/zfsauton/datasets/ArgoRL/swapnilp/new_state_space"]
            self.batch_size = 512
            self.frame_stack = 2
            self.num_workers = 2
            self.train_val_split = 0.95

    # data config
    data_config = TempDataModuleConfig()
    data_module = OfflineCarlaDataModule(data_config)

    # dynamics config
    dyn_ensemble_config = DefaultDynamicsEnsembleConfig()
    dyn_module_config = DefaultDynamicsModuleConfig()
    dynamics = DynamicsEnsemble(
        config=dyn_ensemble_config,
        gpu=dyn_ensemble_config.gpu,
        data_module = data_module,
        state_dim_in = dyn_module_config.state_dim_in,
        state_dim_out = dyn_module_config.state_dim_out,
        action_dim = 2,
        frame_stack = dyn_module_config.frame_stack,
        logger = logger,
        log_freq = 100)


    # Train for 500 epochs
    # dynamics.train(500)


    #############################################################
    #             Test integration of fake env
    ##############################################################
    # env setup (obs, action, reward)
    fake_env_config = DefaultMainConfig()
    fake_env_config.populate_config(\
        obs_config = "DefaultObservationConfig", \
        action_config = "DefaultActionConfig",\
        reward_config="DefaultRewardConfig",\
        uncertainty_config="DefaultUncertaintyConfig")

    env = FakeEnv(dynamics,
                config=fake_env_config,
                logger = logger,
                uncertainty_threshold = 0.5,
                uncertain_penalty = -100,
                timeout_steps = 1,
                uncertainty_params = [0.0045574815320799725, 1.9688976602303934e-05, 0.2866033549975823])

    env.reset()
    env.step(torch.Tensor([-0.5,0.8]))





if __name__ == "__main__":
    main()

