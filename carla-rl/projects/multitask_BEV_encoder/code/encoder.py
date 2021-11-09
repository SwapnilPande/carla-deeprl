'''
Define the encoder class whose object takes BEV stacks as input and outputs their embeddings.
'''
import torch
from torch import nn



class Encoder(nn.Module):

    def __init__(self, params):
        '''
        Arg
            params: A dictionary storing model-type-specific arguments.
        '''
        super(Encoder, self).__init__()

        self.layer_params = params['layer_params']

        if params['model_type'] == 'conv':
            self.mod = self.get_conv_mod()
        else:
            raise Exception ('Invalid model type found.')



    def get_conv_mod(self):
        '''
        Build the convolutional module and return it.
        '''
        pass



    def forward(self, x):

        return self.mod(x)



