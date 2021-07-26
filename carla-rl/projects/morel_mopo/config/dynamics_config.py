# Use Base Config from environment to build config
from environment.config.base_config import BaseConfig
from projects.morel_mopo.algorithm import dynamics_models
from projects.morel_mopo.algorithm import fake_envs

import projects.morel_mopo.algorithm.data_modules as data_modules
import projects.morel_mopo.config.data_module_config as data_module_config



import torch.nn as nn
import torch.optim as optim



class BaseDynamicsConfig(BaseConfig):
    def __init__(self):
        self.gpu = None

        # Which model class to import
        self.dynamics_model_type = None

        # Config for the dynamics model
        self.dynamics_model_config = None

        # Which dataset is associated with the model
        self.dataset_type = None

        # Config for the associated dataset
        self.dataset_config = None

        # Number of epochs to train the dynamics model
        self.train_epochs = None

        # Type of fake env to use with this dynamics model
        self.fake_env_type = None\

    def populate_config(self, gpu = 0):
        self.gpu = gpu
        self.dynamics_model_config.gpu = gpu

        self.verify()


################# MLP #################

class BaseDynamicsModuleConfig(BaseConfig):
    def __init__(self):
        # Dimension of the input to the dynamics
        self.state_dim_in = None

        # Dimension of the dynamics output
        self.state_dim_out = None

        # Frame Stack
        self.frame_stack = None

        # Whether or not to do reward prediction with model
        self.predict_reward = None

        # Number of neurons in each hidden layer of neural net
        self.n_neurons = None

        # Number of shared hidden layers (shared between state prediction and reward heads)
        self.n_hidden_layers = None

        # Number of layers in the prediction heads
        self.n_head_layers = None

        # Dropout probability. If 0, dropout layers are not added
        # Dropout layers are added after each hidden layer
        self.drop_prob = None

        # activation function to use for hidden layers
        self.activation = None

class DefaultDynamicsModuleConfig(BaseConfig):
    def __init__(self):
        self.state_dim_in = 3
        self.state_dim_out = 5
        self.frame_stack = 2
        self.predict_reward = False
        self.n_neurons = 1024
        self.n_hidden_layers = 2
        self.n_head_layers = 2
        self.drop_prob = 0.3
        self.activation = nn.ReLU

class BaseDynamicsEnsembleConfig(BaseConfig):
    def __init__(self):
        # Learning Rate for dynamics ensemble members
        self.lr = None

        # Number of models to use
        self.n_models = None

        # Config for dynamics network
        self.network_cfg = None

        # Which loss function to use
        self.loss = None

        # Dictionary containing any arguments that the loss function needs
        self.loss_args = None

        # Type of optimizer to use
        self.optimizer_type = None

        # Arguments to pass to the optimizer
        self.optimizer_args = None

        # GPU to load model/data on
        self.gpu = None

class DefaultMLPDynamicsEnsembleConfig(BaseConfig):
    def __init__(self):
        self.lr = 0.001
        self.n_models = 5
        self.loss = nn.SmoothL1Loss
        self.loss_args = {"beta" : 0.5}
        self.optimizer_type = optim.Adam
        self.optimizer_args = {"weight_decay" : 0.00}
        self.network_cfg = DefaultDynamicsModuleConfig()
        self.gpu = 2

class DefaultMLPDynamicsConfig(BaseDynamicsConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Which model class to import
        self.dynamics_model_type = dynamics_models.MLPDynamicsEnsemble

        self.dynamics_model_config = DefaultMLPDynamicsEnsembleConfig()

        # Which dataset is associated with the model
        self.dataset_type = data_modules.OfflineCarlaDataModule

        # Config for the associated dataset
        self.dataset_config = data_module_config.MixedDeterministicMLPDataModuleConfig()
        self.dataset_config.frame_stack = self.dynamics_model_config.network_cfg.frame_stack

        self.train_epochs = 200

        self.fake_env_type = fake_envs.FakeEnv


################# Deterministic GRU #################

class BaseGRUDynamicsModuleConfig(BaseConfig):
    def __init__(self):
        self.state_dim_in = None
        self.state_dim_out = None
        self.frame_stack = None
        self.predict_reward = None
        self.gru_input_dim = None
        self.gru_hidden_dim = None
        self.drop_prob = None
        self.activation = None

class DefaultGRUDynamicsModuleConfig(BaseConfig):
    def __init__(self):
        self.state_dim_in = 3
        self.state_dim_out = 5
        self.frame_stack = 50
        self.predict_reward = False
        self.gru_input_dim = 256
        self.gru_hidden_dim = 256
        self.drop_prob = 0.15
        self.activation = nn.ReLU

class DefaultGRUEnsembleDynamicsConfig(BaseConfig):
    def __init__(self):
        self.lr = 0.001
        self.n_models = 5
        self.loss = nn.SmoothL1Loss
        self.loss_args = {"beta" : 0.5, "reduction" : 'none'}
        self.optimizer_type = optim.Adam
        self.network_cfg = DefaultGRUDynamicsModuleConfig()
        self.gpu = 2

class DefaultDeterministicGRUDynamicsConfig(BaseDynamicsConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Which model class to import
        self.dynamics_model_type = dynamics_models.GRUDynamicsEnsemble

        self.dynamics_model_config = DefaultGRUEnsembleDynamicsConfig()

        # Which dataset is associated with the model
        self.dataset_type = data_modules.RNNOfflineCarlaDataModule

        # Config for the associated dataset
        self.dataset_config = data_module_config.MixedDeterministicMLPDataModuleConfig()
        self.dataset_config.frame_stack = self.dynamics_model_config.network_cfg.frame_stack

        self.train_epochs = 200

        self.fake_env_type = fake_envs.RNNFakeEnv


class DefaultProbabilisticGRUDynamicsConfig(BaseDynamicsConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Which model class to import
        self.dynamics_model_type = dynamics_models.ProbabilisticGRUDynamicsEnsemble

        self.dynamics_model_config = DefaultGRUEnsembleDynamicsConfig()

        # Which dataset is associated with the model
        self.dataset_type = data_modules.RNNOfflineCarlaDataModule

        # Config for the associated dataset
        self.dataset_config = data_module_config.MixedDeterministicMLPDataModuleConfig()
        self.dataset_config.frame_stack = self.dynamics_model_config.network_cfg.frame_stack

        self.train_epochs = 200

        self.fake_env_type = fake_envs.RNNFakeEnv


################# Probabilistic MLP #################

class DefaultProbabilisticMLPDynamicsModuleConfig(BaseDynamicsModuleConfig):
    def __init__(self):
        super().__init__()
        self.state_dim_in = 7
        self.state_dim_out = 5
        self.frame_stack = 1
        self.predict_reward = False
        self.n_neurons = 200
        self.n_hidden_layers = 4
        self.n_head_layers = 1
        self.drop_prob = 0
        self.activation = nn.SiLU


class DefaultMLPEnsembleDynamicsConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.n_models = 5
        self.optimizer_type = optim.Adam
        self.network_cfg = DefaultProbabilisticMLPDynamicsModuleConfig()
        self.gpu = None
        self.norm_stats = None


class DefaultProbabilisticMLPDynamicsConfig(BaseDynamicsConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Which model class to import
        self.dynamics_model_type = dynamics_models.ProbabilisticDynamicsEnsemble
        self.dynamics_model_config = DefaultMLPEnsembleDynamicsConfig()

        # Which dataset is associated with the model
        self.dataset_type = data_modules.OfflineCarlaDataModule

        # Config for the associated dataset
        self.dataset_config = data_module_config.MixedProbabilisticMLPDataModuleConfig()
        self.dataset_config.frame_stack = self.dynamics_model_config.network_cfg.frame_stack

        self.train_epochs = 200
        self.fake_env_type = fake_envs.FakeEnv


################# Probabilistic MLP COV #################

class DefaultProbabilisticMLPCovDynamicsModuleConfig(BaseDynamicsModuleConfig):
    def __init__(self):
        super().__init__()
        self.state_dim_in = 7
        self.state_dim_out = 5
        self.frame_stack = 1
        self.predict_reward = False
        self.n_neurons = 200
        self.n_hidden_layers = 4
        self.n_head_layers = 1
        self.drop_prob = 0
        self.activation = nn.SiLU


class DefaultMLPCovEnsembleDynamicsConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.n_models = 5
        self.optimizer_type = optim.Adam
        self.network_cfg = DefaultProbabilisticMLPCovDynamicsModuleConfig()
        self.gpu = None
        self.norm_stats = None


class DefaultProbabilisticMLPDynamicsConfig(BaseDynamicsConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Which model class to import
        self.dynamics_model_type = dynamics_models.ProbabilisticDynamicsEnsemble
        self.dynamics_model_config = DefaultMLPCovEnsembleDynamicsConfig()

        # Which dataset is associated with the model
        self.dataset_type = data_modules.OfflineCarlaDataModule

        # Config for the associated dataset
        self.dataset_config = data_module_config.MixedProbabilisticMLPDataModuleConfig()
        self.dataset_config.frame_stack = self.dynamics_model_config.network_cfg.frame_stack

        self.train_epochs = 200
        self.fake_env_type = fake_envs.FakeEnv


