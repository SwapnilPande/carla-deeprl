'''
Train the BEV image stack encoder with multiple tasks.

Structure Overview
    An encoder is used to encode input BEV stacks into low dimensional embeddings.
    Several branches of model structures follow the embedding in parallel, generating outputs in different formats.
    Multiple learning objectives may share a certain branch, using different channels or dimensions of its output.

Parameters
    encoder_params: A dictionary containing encoder parameters.
        valid keys
            'model_type': The type of the model. One in ['conv'].

            == model_type 'conv' ==
            'layer_params': list of tuples in the following formats
                valid tuples
                    ('conv', filter_num, filter_size, stride, padding)
                    ('max_pool', size, stride, padding)
                    ('glob_avg_pool')
                    ('relu',)

    branch2params: A dictionary with the following items.
        branch_name: params
            valid keys for params
                'type': The type of branch. Valid types are listed below.
                'model_type': The type of the mode. One in ['deconv']
                
                == type 'decoder', model_type 'deconv' ==
                'layer_params': A list of tuples with layer parameters.
                    Valid tuples
                        ('deconv', input_dim_cnt, output_dim_cnt, filt_size, stride, dilation, padding, output_padding)
                        ('relu',)

                == type 'predictor', model_type 'fc' ==
                'layer_params': A list of tuples with layer parameters.
                    Valid tuples
                        ('fc', input_dim_cnt, output_dim_cnt)
                        ('flatten')
                        ('relu',)

    objs: A list of dictionaries (obj_param), each correspond to a learning objective.
        valid keys for obj_param
            'name': A unique name for the learning objective.
            'type': The type of objective. Valid types are listed below.
            'loss_fn': The loss function. One in ['mse']
            'branch': The name of the branch the objective uses.
            'indices': The indices of channels or dimensions of the output of the branch that the objective uses.
            'alpha': The loss weight for its loss term.

            == type reconstruction: Reconstruct the input stack

            == type pred_stack: Predict a stack in the future
            'pred_stack_size': The size of the stack to predict. This determines the output size.
            'pred_time_forward': Number of time steps forward the stack to predict is from now. 1 means the next time step.

            == type curr_value: Estimate a single current value
            'value_key': The key to get the value from json files in the measurement folder.

            == type pred_value: Predict a single future value
            'value_key': The key to get the value from json files in the measurement folder.
            'pred_time_forward': Number of time steps forward the value to predict is from now. 1 means the next time step.

    evals: A list of dictionaries (eval_param), each correspond to an evaluation approach.
        valid keys for eval_param
            'name': A unique name for the evaluation approach.
            'type': The type of evaluation task. Valid types are listed below.
            'metric': The metric that evaluates the performance of the model. One in ['mse']
            
            == type curr_value: Estimate a single current value
            'value_key': The key to get the value from json files in the measurement folder.
            'model':
                The model that takes the embeddings as input and trained on the evaluation task, whose performance reflects the quality of the embedding. One in ['SVR']

            == type pred_value: Predict a single future value
            'value_key': The key to get the value from json files in the measurement folder.
            'pred_time_forward': Number of time steps forward the value to predict is from now. 1 means the next time step.
            'model':
                The model that takes the embeddings as input and trained on the evaluation task, whose performance reflects the quality of the embedding. One in ['SVR']

Note:
    Should support using shared or separated decoders for different objectives.
'''
import torch
import train_fn


from torch.utils.data import DataLoader

from utils import *
from encoder import Encoder
from decoder import Decoder
from predictor import Predictor
from dataset import BEV_Dataset
from train_fn import train_model


def main():

    ####### Use args to store all the parameters and print them in print_params() function.

    '''
    Set parameters
    '''
    device = 'cpu'
    data_dir = '../BEV_data2'

    pix_class_num = 23 # The number of pixel classes after segmentation
    stack_size = 6 # The size of input BEV stack

    # The module to preprocess the data to fit the input of pretrained models
    prepro_mod = nn.Conv3d(pix_class_num, 3, 1).to(device)
    pretrained_name = 'ResNet3D'
    encoder_params = None
    
    branch2params = {\
                'predictor_name': {\
                            'type': 'predictor', \
                            'model_type': 'fc', \
                            'layer_params': [\
                                        ('flatten',), \
                                        ('fc', 512 * 1 * 8 * 8, 512), \
                                        ('relu',), \
                                        ('fc', 512, 2)]}, \

                '3d_decoder': {\
                            'type': 'decoder', \
                            'model_type': 'deconv', \
                            'layer_params': [\
                                        ('deconv', 512, 128, 3, 2, 1, 1, 1), \
                                        ('relu',), \
                                        ('deconv', 128, 64, 3, 2, 1, 1, (0, 1, 1)), \
                                        ('relu',), \
                                        ('deconv', 64, 32, 3, 2, 1, 1, 1), \
                                        ('relu',), \
                                        ('deconv', 32, 2 * pix_class_num, 3, (1, 2, 2), 1, 1, (0, 1, 1))]}}

    objs = [\
        {\
            'name': 'autoencode', \
            'type': 'reconstruction', \
            'loss_fn': 'BCE_w_logits', \
            'branch': '3d_decoder', \
            'indices': (0, pix_class_num), \
            'alpha': 1000.0}, \
        
        # objectives with the same stack size to predict can share a single branch along the channel dimension
        {\
            'name': 'pred_next', \
            'type': 'pred_stack', \
            'loss_fn': 'mse', \
            'branch': '3d_decoder', \
            'indices': (pix_class_num, 2 * pix_class_num), \
            'alpha': 800.0, \
            'pred_stack_size': stack_size, \
            'pred_time_forward': stack_size}, \
        
        {\
            'name': 'est_speed', \
            'type': 'curr_value', \
            'loss_fn': 'mse', \
            'branch': 'predictor_name', \
            'indices': (0, 1), \
            'alpha': 0.3, \
            'value_key': 'speed', \
            }, \

        {\
            'name': 'pred_dist_to_traj', \
            'type': 'pred_value', \
            'loss_fn': 'mse', \
            'branch': 'predictor_name', \
            'indices': (1, 2), \
            'alpha': 0.1, \
            'value_key': 'dist_to_trajectory', \
            'pred_time_forward': 2 * stack_size, \
            }]

    evals = [\
        {\
            'name': 'pred_next_orientation', \
            'type': 'pred_value', \
            'model': 'SVR', \
            'metric': 'mse', \
            'value_key': 'next_orientation', \
            'pred_time_forward': 2 * stack_size}, \

        {\
            'name': 'est_control_steer', \
            'type': 'curr_value', \
            'model': 'SVR', \
            'metric': 'mse', \
            'value_key': 'control_steer'}]

    print_params()
    check_params(branch2params, objs)

    b_size = 32 # batch size
    train_args = {'epochs': 10000, 'lr': 0.001, 'eval_every': 1, 'eval_samps': 100, 'save_every': 20, 'save_name': './debugging_model'}



    '''
    Build dataset and data loader
    '''
    ####### Separate training and validate data either into different folders or implememnt with codes
    train_dataset = BEV_Dataset(data_dir, pix_class_num, stack_size, objs)
    val_dataset = BEV_Dataset(data_dir, pix_class_num, stack_size, evals)
    train_loader = DataLoader(train_dataset, batch_size = b_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = b_size, shuffle = True)

    print ('number of training samples: ', len(val_dataset))
    print ('number of training batches: ', len(train_loader))
    print ('number of validate samples: ', len(val_dataset))
    print ('number of validate batches: ', len(val_loader))



    '''
    Build encoder and decoders (branches)
    '''
    # Build encoder
    if pretrained_name:
        print ('Use a pretrained model as encoder. Variable "encoder_params" is ignored.')
        encoder = get_pretrained(pretrained_name).to(device)
    else:
        raise Exception ('Building models from scratch is not supported yet.')

    # Build branches
    name2branch = dict()
    for name, params in branch2params.items():
        if params['type'] == 'decoder':
            name2branch[name] = Decoder(params).to(device)
        elif params['type'] == 'predictor':
            name2branch[name] = Predictor(params).to(device)
        else:
            raise Exception ('Invalid branch type found.')



    '''
    Train the model
    '''
    train_model(device, train_loader, val_loader, prepro_mod, encoder, name2branch, objs, train_args, branch2params, evals)
    


if __name__ == '__main__':

    main()

