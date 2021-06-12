import sys
import os

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
from projects.morel_mopo.algorithm.fake_env import FakeEnv
import torch

torch.multiprocessing.set_sharing_strategy('file_system') 
EXPERIMENT_NAME = "first_test"
TAGS = ["dyn_only"]

class TempDataModuleConfig():
    def __init__(self):
        self.dataset_paths = ["/zfsauton/datasets/ArgoRL/swapnilp/new_state_space"]
        self.batch_size = 512
        self.frame_stack = 2
        self.num_workers = 1
        self.train_val_split = 0.95


if __name__ == "__main__":
 
    data_config = TempDataModuleConfig()
    data_module = OfflineCarlaDataModule(data_config)

    dyn_config = DefaultDynamicsEnsembleConfig()
    dynamics = DynamicsEnsemble(
        data_module = data_module,
        config = dyn_config,
        # logger = logger
    )
    dynamics.train(500)

