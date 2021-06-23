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
                n_head_layers = 1,
                drop_prob = 0.15,
                activation = "swish"):
        
        #validate inputs:
        assert state_dim_in > 0
        assert state_dim_out > 0
        assert action_dim > 0
        assert n_neurons > 0
        assert drop_prob >= 0 and drop_prob <= 1

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
        layer_list.append(FC(self.input_dim, self.n_neurons, weight_decay=0.000001)
        layer_list.append(self.activation)

        #build hidden layers
        for _ in range(n_hidden_layers):
            layer_list.append(FC(self.n_neurons, self.n_neurons, weight_decay=0.000001))
            layer_list.append(self.activation)

        # Register shared layers by putting them in a module list
        self.shared_layers = nn.ModuleList(layer_list)

        # Next, build the head for the state prediction
        layer_list = []
        # For each hidden layer
        for _ in range(n_head_layers):
            # Add Linear layer
            layer_list.append(nn.Linear(n_neurons, n_neurons))
            # Add activation
            layer_list.append(activation())
            # Add dropout if enabled
            if(drop_prob != 0):
                layer_list.append(nn.Dropout(p = drop_prob))
        layer_list.append(nn.Linear(n_neurons, self.state_dim_out))

        # Register state prediction head layers by putting them in a module list
        self.state_head = nn.ModuleList(layer_list)

        # Build reward head if we're predicting reward
        self.reward_head = None
        if(predict_reward):
            layer_list = []
            # For each hidden layer
            for _ in range(n_head_layers):
                # Add Linear layer
                layer_list.append(nn.Linear(n_neurons, n_neurons))
                # Add activation
                layer_list.append(activation())
                # Add dropout if enabled
                if(drop_prob != 0):
                    layer_list.append(nn.Dropout(p = drop_prob))
            layer_list.append(nn.Linear(n_neurons, self.state_dim_out))

            # Register state prediction head layers by p
            # Register state prediction head layers by putting them in a module list
            self.reward_head = nn.ModuleList(layer_list)
