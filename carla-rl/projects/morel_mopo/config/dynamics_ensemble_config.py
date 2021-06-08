# Use Base Config from environment to build config
from environment.config.base_config import BaseConfig


import torch.nn as nn
import torch.optim as optim



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
        optimizer_type = None

        # GPU to load model/data on
        self.gpu = None

class DefaultDynamicsEnsembleConfig(BaseConfig):
    def __init__(self):
        self.lr = 0.001
        self.n_models = 5
        self.loss = nn.SmoothL1Loss
        self.loss_args = {"beta" : 0.5}
        self.optimizer_type = optim.Adam
        self.network_cfg = DefaultDynamicsModuleConfig()
        self.gpu = 2

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
        self.drop_prob = 0.15
        self.activation = nn.ReLU





