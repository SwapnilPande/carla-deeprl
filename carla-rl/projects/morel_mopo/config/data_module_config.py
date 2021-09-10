
import projects.morel_mopo.algorithm.data_modules as data_modules
from environment.config.base_config import BaseConfig

class BaseDataModuleConfig(BaseConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = None
            self.batch_size = None
            self.frame_stack = None
            self.num_workers = None
            self.train_val_split = None
            self.normalize_data = None
            self.dataset_type = None



# class BaseDeterministicDataModuleConfig(BaseDataModuleConfig):
#     def __init__(self):
#             self.batch_size = 512
#             self.frame_stack = 2
#             self.num_workers = None
#             self.train_val_split = None
#             self.normalize_data = None



class MixedDeterministicMLPDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.frame_stack = 2
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = True
            self.dataset_type = data_modules.OfflineCarlaDataModule

class ObstaclesMixedDeterministicMLPDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_dense_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.frame_stack = 2
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = True
            self.dataset_type = data_modules.OfflineCarlaDataModule


class MixedProbabilisticMLPDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.frame_stack = 5
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = False
            self.dataset_type = data_modules.OfflineCarlaDataModule


class MixedDeterministicGRUDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_regular_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.frame_stack = 50
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = True
            self.dataset_type = data_modules.RNNOfflineCarlaDataModule


class MixedProbabilisticGRUDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.frame_stack = 50
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = False
            self.dataset_type = data_modules.RNNOfflineCarlaDataModule
