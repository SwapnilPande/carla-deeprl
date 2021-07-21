

class BaseDataModuleConfig():
        def __init__(self):
            self.dataset_paths = None
            self.batch_size = None
            self.frame_stack = None
            self.num_workers = None
            self.train_val_split = None
            self.normalize_data = None


class BaseMLPDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = None
            self.batch_size = 512
            self.frame_stack = 2
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = None


class BaseMLPDataModuleConfig(BaseDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = None
            self.batch_size = 512
            self.frame_stack = 50
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = None



# class BaseDeterministicDataModuleConfig(BaseDataModuleConfig):
#     def __init__(self):
#             self.batch_size = 512
#             self.frame_stack = 2
#             self.num_workers = None
#             self.train_val_split = None
#             self.normalize_data = None



class MixedDeterministicMLPDataModuleConfig(BaseMLPDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = True


class MixedProbabilisticMLPDataModuleConfig(BaseMLPDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = False


class MixedDeterministicRNNDataModuleConfig(BaseMLPDataModuleConfig):
        def __init__(self):
            super().__init__()
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
            ]

            self.batch_size = 512
            self.num_workers = 10
            self.train_val_split = 0.95
            self.normalize_data = True