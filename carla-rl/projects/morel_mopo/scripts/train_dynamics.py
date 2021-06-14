import sys
import os
import torch
# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

# from common.loggers.comet_logger import CometLogger
# from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, BaseDynamicsEnsembleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
from projects.morel_mopo.algorithm.fake_env import FakeEnv
EXPERIMENT_NAME = "first_test"
TAGS = ["dyn_only"]

def main():
    # logger_conf = CometLoggerConfig()
    # logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = TAGS)


    # logger = CometLogger(logger_conf)

    class TempDataModuleConfig():
        def __init__(self):
            self.dataset_paths = ["/zfsauton/datasets/ArgoRL/swapnilp/new_state_space"]
            self.batch_size = 512
            self.frame_stack = 2
            self.num_workers = 2
            self.train_val_split = 0.95

    data_config = TempDataModuleConfig()
    data_module = OfflineCarlaDataModule(data_config)

    dyn_config = DefaultDynamicsEnsembleConfig()

    dynamics = DynamicsEnsemble(
        data_module = data_module,
        config = dyn_config,
        # logger = logger
    )


    # Train for 500 epochs
    dynamics.train(500)

    # env = FakeEnv(dynamics,
    #             dyn_config=dyn_config,
    #             logger = None,
    #             uncertainty_threshold = 0.5,
    #             uncertain_penalty = -100,
    #             timeout_steps = 1,
    #             uncertainty_params = [0.0045574815320799725, 1.9688976602303934e-05, 0.2866033549975823])

    # env.reset()
    # env.step(torch.Tensor([-0.5,0.8]))




if __name__ == "__main__":
    main()

