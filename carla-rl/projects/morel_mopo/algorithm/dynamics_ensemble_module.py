
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import scipy.spatial
import os
from tqdm import tqdm
import time

class DynamicsMLP(nn.Module):
    def __init__(self,
                state_dim_in,
                state_dim_out,
                action_dim,
                frame_stack,
                predict_reward = False,
                n_neurons = 1024,
                n_hidden_layers = 4,
                n_head_layers = 1,
                drop_prob = 0.15,
                activation = nn.ReLU):
        super(DynamicsMLP, self).__init__()

        # Validate inputs
        assert state_dim_in > 0
        assert state_dim_out > 0
        assert action_dim > 0
        assert n_neurons > 0
        assert drop_prob >= 0 and drop_prob <= 1

        # Store configuration parameters

        # Input is the input state plus actions state
        self.input_dim = frame_stack*(state_dim_in + action_dim)
        # Output is the output state dim + reward if True
        self.output_dim = state_dim_out + int(predict_reward)
        self.state_dim_out = state_dim_out

        self.n_neurons = n_neurons

        # Dynamically construct nn according to arguments
        layer_list = []
        # First add the input layer and activation
        layer_list.append(nn.Linear(self.input_dim, n_neurons))
        layer_list.append(activation())

        # For each hidden layer
        for _ in range(n_hidden_layers):
            # Add Linear layer
            layer_list.append(nn.Linear(n_neurons, n_neurons))
            # Add activation
            layer_list.append(activation())
            # Add dropout if enabled
            if(drop_prob != 0):
                layer_list.append(nn.Dropout(p = drop_prob))

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



    def forward(self, x):
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)

        # State Head
        # Apply first layer
        output = self.state_head[0](x)
        # Apply all other layers
        for layer in self.state_head[1:]:
            output = layer(output)

        # Apply reward head if we are predicting rewards
        if self.reward_head is not None:
            reward_out = self.reward_head[0](x)
            for layer in self.reward_head[1:]:
                reward_out = layer(reward_out)
            output = torch.cat([output, reward_out], dim = 1)

        return output

class DynamicsEnsemble:
    def __init__(self,
                    data_module,
                    config,

                    logger = None,
                    log_freq = 100):

        self.config = config

        if self.config.gpu == -1:
            self.device = "cpu"
        else:
            self.device = "cuda:{}".format(self.config.gpu)

        self.logger = logger
        self.log_freq = log_freq

        # TODO LOG ALL HYPERPARAMETERS
        # Save Parameters
        # Setup data module
        self.data_module = data_module
        self.data_module.setup()

        # Model parameters
        self.n_models = config.n_models
        self.network_cfg = config.network_cfg
        # Get the shape of the input, output, and frame stack from the data module
        self.state_dim_in = self.data_module.state_dim_in
        self.state_dim_out = self.data_module.state_dim_out
        self.action_dim = self.data_module.action_dim
        self.frame_stack = self.data_module.frame_stack

        self.lr = config.lr
        # Build the loss function using loss args
        self.loss = config.loss(**config.loss_args)

        # Validation loss
        self.mse_loss = torch.nn.MSELoss()

        # Save optimizer object
        self.optimizer_type = config.optimizer_type

        # Create n_models models
        self.models = nn.ModuleList()
        for i in range(self.n_models):
            self.models.append(DynamicsMLP(
                state_dim_in = self.state_dim_in,
                state_dim_out = self.state_dim_out,
                action_dim = self.action_dim,
                frame_stack = self.frame_stack,
                predict_reward = self.network_cfg.predict_reward,
                n_neurons = self.network_cfg.n_neurons,
                n_hidden_layers = self.network_cfg.n_hidden_layers,
                n_head_layers = self.network_cfg.n_head_layers,
                drop_prob = self.network_cfg.drop_prob,
                activation = self.network_cfg.activation
            ))
            self.models[-1].to(self.device)

    def get_input_output_dim(self):
        return self.state_dim_in, self.state_dim_out

    def get_gpu(self):
        return self.config.gpu



    def get_data_module(self):
        return self.data_module

    def get_n_models(self):
        return self.n_models

    def to(self, device):
        for i in range(self.n_models):
            self.models[i].to(device)

    def forward(self, x, model_idx = None):
        return self.models[model_idx](x)


    def prepare_batch(self, state_in, actions, delta, reward):
        # Input should be obs and actions concatenated, and flattened

        # state_in is [batch_size, frame_stack, state_dim]
        # action is [batch_size, frame_stack, action_dim]
        # feed tensor is [batch_size, frame_stack * (state_dim + action_dim)]

        # Conver to cuda first
        state_in = state_in.to(self.device)
        actions = actions.to(self.device)
        delta = delta.to(self.device)
        # reward = reward.to(self.device)

        feed_tensor = torch.reshape(
                            torch.cat([state_in, actions], dim = 2),
                                (-1, self.frame_stack * (self.state_dim_in + self.action_dim)
                            )
                      )
        # Concatenate delta and reward if we are predicting reward
        if(self.network_cfg.predict_reward):
            return feed_tensor, torch.cat([delta,reward], dim = 1)

        # Else, just return delta
        return feed_tensor, delta


    def training_step(self, batch, model_idx):
        # Zero Optimizer gradients
        self.optimizers[model_idx].zero_grad()

        # Split batch into componenets
        obs, actions, rewards, delta, done, waypoints, num_wps_list, vehicle_pose = batch
        obs = (torch.stack(obs)).to(self.device)
        actions = (torch.stack(actions)).to(self.device)
        delta = (torch.stack(delta)).to(self.device)
        # waypoints = (torch.stack(waypoints))to(self.device)
        vehicle_pose = (torch.stack(vehicle_pose)).to(self.device)

        # Combine tensors and reshape batch to flat inputs
        feed, target = self.prepare_batch(obs, actions, delta, rewards)

        # Make prediction with selected model
        y_hat = self.forward(feed, model_idx = model_idx)

        # Compute loss
        loss = self.loss(y_hat, target)

        # Backpropagate
        loss.backward()

        # Take optimizer step
        self.optimizers[model_idx].step()

        # Return metrics to log
        return {"model_{}_loss".format(model_idx) : loss}


    def validation_step(self, batch, model_idx = 0):

        # Split batch into componenets
        obs, actions, rewards, delta, done, waypoints, num_wps_list, vehicle_pose = batch
        obs = (torch.stack(obs)).to(self.device)
        actions = (torch.stack(actions)).to(self.device)
        delta = (torch.stack(delta)).to(self.device)
        # waypoints = (torch.stack(waypoints))to(self.device)
        vehicle_pose = (torch.stack(vehicle_pose)).to(self.device)
        # Combine tensors and reshape batch to flat inputs
        feed, target = self.prepare_batch(obs, actions, delta, rewards)

        # Predictions by each model
        y_hat = self.forward(feed, model_idx = model_idx)

        # Compute loss
        loss = self.loss(y_hat, target)
        mse_loss = self.mse_loss(y_hat, target)

        return {"model_{}_val_loss".format(model_idx): loss,
                "model_{}_val_mse_loss".format(model_idx) : mse_loss}


    def log_metrics(self, epoch, batch_idx, num_batches, metric_dict):
        if(self.logger is not None):
            step = num_batches*epoch + batch_idx

            for metric_name, value in metric_dict.items():
                self.logger.log_scalar(metric_name, value, step)

    def train(self, epochs):
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()

        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)

        # # Log hyperparameters
        # self.logger.log_hyperparameters({
        #     "dyn_ens/lr" : self.lr,
        #     "dyn_ens/optimizer" : str(self.optimizer_type),
        #     "dyn_ens/n_models" : self.n_models,
        #     "dyn_ens/batch_size" : train_dataloader.batch_size,
        #     "dyn_ens/train_val_split" : self.data_module.train_val_split,
        #     "dyn_ens/epochs" : epochs,
        #     "dyn_ens/predict_reward" : self.network_cfg.predict_reward,
        #     "dyn_ens/n_neurons" : self.network_cfg.n_neurons,
        #     "dyn_ens/n_hidden_layers" : self.network_cfg.n_hidden_layers,
        #     "dyn_ens/n_head_layers" : self.network_cfg.n_head_layers,
        #     "dyn_ens/drop_prob" : self.network_cfg.drop_prob,
        #     "dyn_ens/activation" : str(self.network_cfg.activation)

        # })


        # Configure the optimizers
        self.optimizers = self.configure_optimizers()

        # Only run validation if val dataloader has elements
        if(len(val_dataloader) <= 0):
            print("WARNING: Skipping validation since validation dataloader has 0 elements.")

        for epoch in range(epochs): # Loop over epochs
            time.sleep(2)
            for model_idx in range(self.n_models): # Loop over models

                with tqdm(total = num_train_batches) as pbar:
                    pbar.set_description_str("Train epoch {}, Model {}:".format(epoch, model_idx))
                   
           
                    for batch_idx, batch in enumerate(train_dataloader): # Loop over batches
                        # Run training step for jth model
                        log_params = self.training_step(batch, model_idx)

                        pbar.set_postfix_str("loss: {}".format(log_params['model_{}_loss'.format(model_idx)]))
                        pbar.update(1)

                        if batch_idx % self.log_freq == 0 and self.logger is not None:
                            self.log_metrics(epoch, batch_idx, num_train_batches, log_params)


                with tqdm(total = num_val_batches) as pbar:
                    pbar.set_description_str("Validation:".format(epoch))
                    for batch_idx, batch in enumerate(val_dataloader): # Loop over batches
                        # Run training step for jth model
                        log_params = self.validation_step(batch, model_idx)

                        pbar.set_postfix_str("epoch {}, model_idx: {}, loss: {}".format(epoch, model_idx, log_params['model_{}_val_loss'.format(model_idx)]))
                        pbar.update(1)

                        if batch_idx % self.log_freq == 0 and self.logger is not None:
                            self.log_metrics(epoch, batch_idx, num_val_batches, log_params)

    def configure_optimizers(self):
        # Define optimizers
        # self.optimizer is defined in __init__
        # This is the type of optimizer we want to use
        optimizers = [self.optimizer_type(model.parameters()) for model in self.models]

        return optimizers

    # def on_train_epoch_end(self, outputs):
    #     self.model_idx += 1

    #     if(self.model_idx == self.n_models):
    #         self.model_idx = 0

    #     self.log('selected_model', self.model_idx, prog_bar = True)


    # def on_validation_epoch_end(self):
    #     val_data = torch.tensor(self.val_data)
    #     self.val_data = None

    #     is_nan = torch.isnan(val_data)
    #     means = torch.mean(val_data, dim = 0)

    #     if(self.trainer.train_dataloader is not None):
    #         # Actual epoch, when adjusted for number of models
    #         actual_epoch = self.trainer.current_epoch // self.n_models

    #         log_step = actual_epoch * len(self.trainer.train_dataloader)

    #     else:
    #         log_step = 0

    #     self.logger.log_metrics({'val_loss'.format(self.model_idx) : means[0]},
    #             step = log_step)

    #     self.logger.log_metrics({'val_mse'.format(self.model_idx) : means[1]},
    #             step = log_step)

    #     # self.logger.log_metrics({'val_rel_error'.format(self.model_idx) : means[2]},
    #     #         step = log_step)

    #     # self.logger.log_metrics({'max_val_rel_error'.format(self.model_idx) : means[3]},
    #     #         step = log_step)

    #     # self.logger.log_metrics({'min_val_rel_error'.format(self.model_idx) : means[4]},
    #     #         step = log_step)


    #     # self.model_idx = self.train_model_idx

    def configure_optimizers(self):
        # Define optimizers
        # self.optimizer is defined in __init__
        # This is the type of optimizer we want to use
        optimizers = [self.optimizer_type(model.parameters()) for model in self.models]

        return optimizers

    # def get_input_output_dim(self):
    #     return self.models[0].input_dim, self.models[0].output_dim

    # def predict(self, x):
    #     # Generate prediction of next state using dynamics model
    #     with torch.set_grad_enabled(False):
    #         return torch.stack(list(map(lambda i: self.forward(x, model_idx = i), range(self.n_models))))

    # def optimizer_step(self,
    #                     current_epoch,
    #                     batch_nb,
    #                     optimizer,
    #                     optimizer_i,
    #                     second_order_closure,
    #                     on_tpu,
    #                     using_native_amp,
    #                     using_lbfgs):
    #     """Override default step method to train one model per epoch
    #     """

    #     if(current_epoch % self.n_models == optimizer_i):
    #         optimizer.step()
    #         optimizer.zero_grad()






    # def train(self, dataloader, epochs = 5, optimizer = torch.optim.Adam, loss = nn.MSELoss, summary_writer = None, comet_experiment = None):

    #     hyper_params = {
    #         "dynamics_n_models":  self.n_models,
    #         "usad_threshold": self.threshold,
    #         "dynamics_epochs" : 5
    #     }
    #     if(comet_experiment is not None):
    #         comet_experiment.log_parameters(hyper_params)

    #     # Define optimizers and loss functions
    #     self.optimizers = [None] * self.n_models
    #     self.losses = [None] * self.n_models

    #     for i in range(self.n_models):
    #         self.optimizers[i] = optimizer(self.models[i].parameters())
    #         self.losses[i] = loss()

    #     # Start training loop
    #     for epoch in range(epochs):
    #         for i, batch in enumerate(tqdm(dataloader)):
    #             # Split batch into input and output
    #             feed, target = batch

    #             loss_vals = list(map(lambda i: self.train_step(i, feed, target), range(self.n_models)))

    #             # Tensorboard
    #             if(summary_writer is not None):
    #                 for j, loss_val in enumerate(loss_vals):
    #                     summary_writer.add_scalar('Loss/dynamics_{}'.format(j), loss_val, epoch*len(dataloader) + i)

    #             if(comet_experiment is not None and i % 10 == 0):
    #                 for j, loss_val in enumerate(loss_vals):
    #                     comet_experiment.log_metric('dyn_model_{}_loss'.format(j), loss_val, epoch*len(dataloader) + i)
    #                     comet_experiment.log_metric('dyn_model_avg_loss'.format(j), sum(loss_vals)/len(loss_vals), epoch*len(dataloader) + i)


    # def usad(self, predictions):
    #     # Compute the pairwise distances between all predictions
    #     distances = scipy.spatial.distance_matrix(predictions, predictions)

    #     # If maximum is greater than threshold, return true
    #     return (np.amax(distances) > self.threshold)

    # def save(self, save_dir):
    #     for i in range(self.n_models):
    #         torch.save(self.models[i].state_dict(), os.path.join(save_dir, "dynamics_{}.pt".format(i)))

    # def load(self, load_dir):
    #     for i in range(self.n_models):
    #         self.models[i].load_state_dict(torch.load(os.path.join(load_dir, "dynamics_{}.pt".format(i)), map_location=self.device))

