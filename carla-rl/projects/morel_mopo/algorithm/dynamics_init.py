import torch.nn as nn
import torch.nn.functional as F


'''
This file is modified from https://github.com/Auton-Self-Driving/carla-deeprl/blob/offline-mbrl/carla-rl/projects/morel_mopo/algorithm/dynamics_ensemble_module.py
 and https://github.com/JannerM/mbpo/blob/master/mbpo/models/fc.py
We are using the structure of morel-mopo, 
while instantiating the model as in the second link
'''



##################
# One Layer 
##################

'''
DynamicsFC instantiate one layer in the dynamics model
dim_in: input dimension
dim_out: output dimension
activation(str): currently using swish activation 
n_neurons: size of the layer
weight_decay: rate of weight decay
bias
'''
class FC(nn.Module):

    def __init__(self, dim_in, dim_out, weight_decay=None, bias=None):
        super(FC, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight_decay = weight_decay
        self.bias = bias
        self.weight = nn.Parameter(torch.Tensor(dim_out, dim_in))
    
    def forward(self, x):
        if self.bias: return torch.dot(self.weight,x) + self.bias
        else: return torch.dot(self.weight,x)



##################
# One Model 
##################
'''
Input:
    drop_prob: disabled
    activation: default is swish
functionality:
    __init__: initialize
    forward: forward within one model
    loss: TBD

'''


class OneModel(nn.Module):

    _activations = {
        None: nn.SiLU,
        "swish": nn.SiLU,
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax
    }

    def __init__(self,
                state_dim_in,
                state_dim_out,
                action_dim,
                frame_stack,
                predict_reward = False,
                n_neurons = 200,
                n_hidden_layers = 4,
                activation = "swish"):
        
        #validate inputs:
        assert state_dim_in > 0
        assert state_dim_out > 0
        assert action_dim > 0
        assert n_neurons > 0

        #initialization
        super(OneModel, self).__init__()
        self.activation = _activations[activation]
        self.input_dim = frame_stack*(state_dim_in + action_dim)
        self.output_dim = state_dim_out + int(predict_reward)
        self.state_dim_in = state_dim_in
        self.state_dim_out = state_dim_out
        self.n_neurons = n_neurons


        # Dynamically construct nn according to arguments
        layer_list = []

        #input layer and activation
        layer_list.append(FC(self.input_dim, self.n_neurons, weight_decay=0.000001))
        layer_list.append(self.activation)

        #build hidden layers
        for _ in range(n_hidden_layers):
            layer_list.append(FC(self.n_neurons, self.n_neurons, weight_decay=0.000001))
            layer_list.append(self.activation)

        #build output layer
        layer_list.append(FC(self.n_neurons, self.output_dim, weight_decay=0.000001))

        # Register shared layers by putting them in a module list
        self.shared_layers = nn.ModuleList(layer_list)


    def forward(self, x):
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)
        return x



##################
# Dynamics Model 
##################
'''
Input:
    drop_prob: disabled
    activation: default is swish
functionality:
    __init__: initialize
    forward: return the mean and variance of predictions all models when model_idx is None
            otherwise, return the prediction of one specified model

'''
class DynamicsModel(nn.Module):
    def __init__(self,
                config,
                data_module = None,
                state_dim_in = None,
                state_dim_out = None,
                action_dim = None,
                predict_reward = 1,
                frame_stack = None,
                norm_stats = None,
                gpu = None,
                logger = None,
                log_freq = 100):
        super(DynamicsModel, self).__init__()
        
        # initialization
        if data_module:
            self.data_module = data_module
            self.state_dim_in = self.data_module.state_dim_in
            self.state_dim_out = self.data_module.state_dim_out
            self.action_dim = self.data_module.action_dim
            self.frame_stack = self.data_module.frame_stack

        else:
            self.data_module = None
            self.state_dim_in = state_dim_in
            self.state_dim_out = state_dim_out
            self.action_dim = action_dim
            self.frame_stack = frame_stack
        
        #initialize variables according to paper
        self.n_neurons = 200
        self.n_hidden_layers = 4
        self.activation = "swish"
        self.predict_reward = predict_reward

            
        # Build n models
        self.models = nn.ModuleList()
        for i in range(self.n_models):
            self.models.append(OneModel(
                state_dim_in = self.state_dim_in,
                state_dim_out = self.state_dim_out,
                action_dim = self.action_dim,
                frame_stack = self.frame_stack,
                predict_reward = self.predict_reward,
                n_neurons = self.n_neurons,
                n_hidden_layers = self.n_hidden_layers,
                activation = self.activation
            ))
            self.models[-1].to(self.device)


    
    def forward(self, x, model_idx=None):
        if model_idx is None:
            # predict for all models
            predictions = [model(x) for model in self.models]
            mean = sum(predictions)/self.n_models
            var = sum((i - mean) ** 2 for i in predictions) / self.n_models
            return mean, var
        else:
            # predict for specified model
            predictions = self.models[i](x)
            return predictions





        
