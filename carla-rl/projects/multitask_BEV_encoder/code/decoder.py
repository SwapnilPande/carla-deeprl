'''
Define a decoder class whose object takes a BEV embedding as input and outputs a tensor with the same shape of the BEV stack, possibly with different number of channels.
This decoder can be used for various learning objectives, such as input reconstruction or future BEV stack prediction.
'''
import torch
from torch import nn



class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()

        self.layer_params = params['layer_params']

        if params['model_type'] == 'deconv':
            self.mod = self.get_deconv_mod()
        else:
            raise Exception ('Invalid model_type found.')


    def get_deconv_mod(self):

        mods = list()

        for layer_param in self.layer_params:
            layer_type = layer_param[0]
            
            if layer_type == 'deconv':
                in_channel_cnt = layer_param[1]
                out_channel_cnt = layer_param[2]
                filt_size = layer_param[3]
                stride = layer_param[4]
                dilation = layer_param[5]
                padding = layer_param[6]
                output_padding = layer_param[7]
                mods.append(nn.ConvTranspose3d(in_channel_cnt, out_channel_cnt, filt_size, stride = stride, dilation = dilation, padding = padding, output_padding = output_padding))

            elif layer_type == 'relu':
                mods.append(nn.ReLU())

            else:
                raise Exception ('Invalid layer_type found.')

        return nn.Sequential(*mods)


    def forward(self, x):
        
        return self.mod(x)



