'''
Define a Predictor class whose object takes a BEV embedding as input and outputs a vector that predicts various variables.
'''
import torch
from torch import nn



class Predictor(nn.Module):

    def __init__(self, params):
        super(Predictor, self).__init__()

        self.layer_params = params['layer_params']

        if params['model_type'] == 'fc':
            self.mod = self.get_fc_mod()
        else:
            raise Exception ('Invalid model_type found.')


    def get_fc_mod(self):

        mods = list()

        for layer_param in self.layer_params:
            layer_type = layer_param[0]
            
            if layer_type == 'flatten':
                mods.append(nn.Flatten())

            elif layer_type == 'fc':
                in_channel_cnt = layer_param[1]
                out_channel_cnt = layer_param[2]
                mods.append(nn.Linear(in_channel_cnt, out_channel_cnt))

            elif layer_type == 'relu':
                mods.append(nn.ReLU())

            else:
                raise Exception ('Invalid layer_type found.')

        return nn.Sequential(*mods)


    def forward(self, x):
        
        return self.mod(x)



