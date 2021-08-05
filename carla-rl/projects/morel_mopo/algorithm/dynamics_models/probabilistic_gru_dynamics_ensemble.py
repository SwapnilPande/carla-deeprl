import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import scipy.spatial
import os
from tqdm import tqdm

class MaskedNLLLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(MaskedNLLLoss, self).__init__()

    def forward(self, input, target, mask):

        # Get neg_log_prob for all batch dim, time step, state dim
        neg_log_prob = - input.log_prob(target)
        # Average across state dim
        mean_neg_log_prob = torch.sum(neg_log_prob, dim = -1) / neg_log_prob.shape[-1]

        # Apply mask
        masked = mask * mean_neg_log_prob

        # Average across all other dimensions
        mean_loss = torch.sum(masked) / torch.sum(mask)

        return mean_loss


class ProbabilisticDynamicsGRU(nn.Module):
    def __init__(self,
                state_dim_in,
                state_dim_out,
                action_dim,
                predict_reward = False,
                gru_input_dim = 256,
                gru_hidden_dim = 256,
                drop_prob = 0.15,
                activation = nn.ReLU):
        super(ProbabilisticDynamicsGRU, self).__init__()

        # Validate inputs
        assert state_dim_in > 0
        assert state_dim_out > 0
        assert action_dim > 0
        assert drop_prob >= 0 and drop_prob <= 1

        # Store configuration parameters
        # Input is the input state plus actions state
        self.input_dim = state_dim_in + action_dim
        # Output is the output state dim + reward if True
        # Multiply dimension by 2 to predict mean, variance
        self.predict_reward = predict_reward
        self.output_dim = state_dim_out + int(predict_reward)
        self.state_dim_out = state_dim_out

        self.gru_input_dim = gru_input_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Input MLP layer, encode input into GRU dimension
        self.input_layer = nn.Linear(self.input_dim, self.gru_input_dim)
        self.input_act = activation()


        # Construct GRU
        # Construct GRU
        self.gru = nn.GRU(input_size = self.gru_input_dim,
                            hidden_size = self.gru_hidden_dim,
                            batch_first = True)

        # Output MLP layer, change to output dim
        self.mean_state_head = nn.Linear(self.gru_hidden_dim, self.output_dim)
        self.std_state_head = nn.Linear(self.gru_hidden_dim, self.output_dim)
        self.std_act = torch.exp

        # Build reward head if we're predicting reward
        self.reward_head = None
        if(predict_reward):
            self.reward_head = nn.Linear(self.gru_hidden_dim, 1)

    def forward(self, x, hidden_state = None):
        # x = torch.flatten(x)
        # Shared layers
        x = self.input_layer(x)
        x = self.input_act(x)
        x, hidden_state = self.gru(x, hx = hidden_state)

        # Compute mean and std, return values concatenated
        mean = self.mean_state_head(x)
        std = self.std_state_head(x)
        std = self.std_act(std)

        # Apply reward head if we are predicting rewards
        reward_out = None
        if self.reward_head is not None:
            reward_out = self.reward_head(x)

        return mean, std, reward_out, hidden_state

class ProbabilisticGRUDynamicsEnsemble(nn.Module):
    log_dir = "dynamics_ensemble"
    model_log_dir = os.path.join(log_dir, "models")

    def __init__(self,
                    config,

                    data_module = None,
                    state_dim_in = None,
                    state_dim_out = None,
                    action_dim = None,
                    frame_stack = None,
                    norm_stats = None,
                    gpu = None,

                    logger = None,
                    log_freq = 500,
                    disable_bars = True):
        super(ProbabilisticGRUDynamicsEnsemble, self).__init__()

        self.config = config

        # Save the logger
        self.logger = logger

        self.disable_bars = disable_bars

        # Save config to load in the future
        if logger is not None:
            self.logger.pickle_save(self.config, ProbabilisticGRUDynamicsEnsemble.log_dir, "config.pkl")


        self.log_freq = log_freq

        # Save Parameters

        # This module can be setup by passing a data module or by passing the shape of the
        # output separately
        self.data_module = data_module
        if(self.data_module is not None):
            # Setup data module
            self.data_module = data_module
            self.data_module.setup()
            # Get the shape of the input, output, and frame stack from the data module
            self.state_dim_in = self.data_module.state_dim_in
            self.state_dim_out = self.data_module.state_dim_out
            self.action_dim = self.data_module.action_dim
            self.frame_stack = self.data_module.frame_stack
            self.normalization_stats = self.data_module.normalization_stats
        else:
            # Validate that all of these inputs are passed
            self.state_dim_in = state_dim_in
            if(self.state_dim_in is None):
                raise Exception("state_dim_in is None, this must be passed if data_module is None")

            self.state_dim_out = state_dim_out
            if(self.state_dim_out is None):
                raise Exception("state_dim_out is None, this must be passed if data_module is None")

            self.action_dim = action_dim
            if(self.action_dim is None):
                raise Exception("action_dim is None, this must be passed if data_module is None")

            self.frame_stack = frame_stack
            if(self.frame_stack is None):
                raise Exception("frame_stack is None, this must be passed if data_module is None")

            self.normalization_stats = norm_stats
            if(self.normalization_stats is None):
                raise Exception("norm_stats is None, this must be passed if data_module is None")

            self.gpu = gpu
            if(self.gpu is None):
                raise Exception("gpu is None, this must be passed if data_module is None")

        if logger is not None:
            state_params = (self.state_dim_in,
                            self.state_dim_out,
                            self.action_dim,
                            self.frame_stack,
                            self.normalization_stats)
            self.logger.pickle_save(state_params, ProbabilisticGRUDynamicsEnsemble.log_dir, "state_params.pkl")

        # Model parameters
        self.n_models = config.n_models
        self.network_cfg = config.network_cfg

        self.lr = config.lr
        # Build the loss function using loss args
        #TODO FIX THIS
        self.loss = MaskedNLLLoss() #config.loss(**config.loss_args)

        # Save optimizer object
        self.optimizer_type = config.optimizer_type

        # Setup GPU
        self.gpu = self.config.gpu
        if self.gpu == -1:
            self.device = "cpu"
        else:
            self.device = "cuda:{}".format(self.gpu)

        # Create n_models models
        self.models = nn.ModuleList()
        for i in range(self.n_models):
            self.models.append(ProbabilisticDynamicsGRU(
                state_dim_in = self.state_dim_in,
                state_dim_out = self.state_dim_out,
                action_dim = self.action_dim,
                predict_reward = self.network_cfg.predict_reward,
                gru_input_dim = self.network_cfg.gru_input_dim,
                gru_hidden_dim = self.network_cfg.gru_hidden_dim,
                drop_prob = self.network_cfg.drop_prob,
                activation = self.network_cfg.activation
            ))
            self.models[-1].to(self.device)

    def forward(self, x, model_idx=None, hidden_state = None, return_dist = False):
        # Create a list of the models to iterate over
        if model_idx is None:
            model_indices = list(range(self.n_models))
        else:
            model_indices = [model_idx]

        # If the hidden state is None, then create an array of hidden states to pass to the model
        if(hidden_state is None):
            hidden_state = [None] * len(model_indices)

        state_predictions = []
        reward_predictions = []
        hidden_states = []
        # predict for all models
        for model_i, h in zip(model_indices, hidden_state):
            # Get output from model
            mean, std, reward, h = self.models[model_i](x, hidden_state = h)

            # Build diagonal gaussian model
            dist = torch.distributions.normal.Normal(
                            loc = mean,
                            scale = std
                        )

            # If not return distribtuon, then sample from the distribution
            if(not return_dist):
                output = dist.sample()
            # Return (dist, reward)
            else:
                output = dist

            state_predictions.append(output)
            reward_predictions.append(reward)
            hidden_states.append(h)

        # if(len(predictions) == 1):
        #     predictions = predictions[0]

        return (state_predictions, reward_predictions), hidden_states

    def prepare_feed(self, state_in, actions):
        # Convert to cuda first
        state_in = state_in.to(self.device)
        actions = actions.to(self.device)

        return torch.cat([state_in, actions], dim = -1)

    def prepare_target(self, delta, reward):
        delta = delta.to(self.device)
        reward = reward.to(self.device)

        if(self.network_cfg.predict_reward):
            return torch.cat([delta,reward], dim = -1)

        return delta

    def prepare_batch(self, state_in, actions, delta, reward, mask):
        # Input should be obs and actions concatenated, and flattened

        # state_in is [batch_size, frame_stack, state_dim]
        # action is [batch_size, frame_stack, action_dim]
        # feed tensor is [batch_size, frame_stack * (state_dim + action_dim)]

        feed_tensor = self.prepare_feed(state_in, actions)

        target_tensor = self.prepare_target(delta, reward)

        # Else, just return delta
        return feed_tensor, target_tensor, mask.to(self.device)


    def training_step(self, batch, model_idx):
        # Zero Optimizer gradients
        self.optimizers[model_idx].zero_grad()

        # Split batch into componenets
        obs, actions, rewards, delta, done, vehicle_pose, mask = batch

        # print('obs', obs.shape, obs)
        # print('actions', actions.shape, actions)
        # print('rew', rewards.shape, rewards)
        # print('delt', delta.shape, delta)
        # print('done', done.shape, done)
        # print('vehpose', vehicle_pose.shape, vehicle_pose)

        # Combine tensors and reshape batch to flat inputs
        feed, target, mask = self.prepare_batch(obs, actions, delta, rewards, mask)

        # Make prediction with selected model
        (y_hat, reward), hidden_state = self.forward(feed, model_idx = model_idx, return_dist = True)
        # Extract prediction from list
        y_hat = y_hat[0]

        # Compute loss
        loss = self.loss(y_hat, target, mask)

        # Backpropagate
        loss.backward()

        # Take optimizer step
        self.optimizers[model_idx].step()

        # Return metrics to log
        return {"model_{}_loss".format(model_idx) : loss}


    def validation_step(self, batch, model_idx = 0):
        with torch.no_grad():
            # Split batch into componenets
            obs, actions, rewards, delta, done, vehicle_pose, mask = batch

            # Combine tensors and reshape batch to flat inputs
            feed, target, mask = self.prepare_batch(obs, actions, delta, rewards, mask)

            # Predictions by each model
            (y_hat, reward), hidden_state = self.forward(feed, model_idx = model_idx, return_dist = True)
            # Extract prediction from list
            y_hat = y_hat[0]

            # Compute loss
            loss = self.loss(y_hat, target, mask)

            return {"model_{}_val_loss".format(model_idx): loss}


    def log_metrics(self, epoch, batch_idx, num_batches, metric_dict):
        if(self.logger is not None):
            step = num_batches*epoch + batch_idx

            for metric_name, value in metric_dict.items():
                self.logger.log_scalar(metric_name, value, step)

    def train_model(self, epochs, n_incremental_models = 10):
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()

        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)

        # Log hyperparameters
        if self.logger is not None:
            self.logger.log_hyperparameters({
                "dyn_ens/lr" : self.lr,
                "dyn_ens/optimizer" : str(self.optimizer_type),
                "dyn_ens/n_models" : self.n_models,
                "dyn_ens/batch_size" : train_dataloader.batch_size,
                "dyn_ens/train_val_split" : self.data_module.train_val_split,
                "dyn_ens/epochs" : epochs,
                "dyn_ens/predict_reward" : self.network_cfg.predict_reward,
                "dyn_ens/gru_input_dim" : self.network_cfg.gru_input_dim,
                "dyn_ens/gru_hidden_dim" : self.network_cfg.gru_hidden_dim,
                "dyn_ens/drop_prob" : self.network_cfg.drop_prob,
                "dyn_ens/activation" : str(self.network_cfg.activation),
                "dyn_ens/network_type" : "Probabilistic GRU"

            })


        # Configure the optimizers
        self.optimizers = self.configure_optimizers()

        # Only run validation if val dataloader has elements
        if(len(val_dataloader) <= 0):
            print("WARNING: Skipping validation since validation dataloader has 0 elements.")


        steps_between_model_save = epochs // n_incremental_models

        for epoch in range(epochs): # Loop over epochs
            for model_idx in range(self.n_models): # Loop over models

                self.train()
                with tqdm(total = num_train_batches, disable = self.disable_bars) as pbar:
                    pbar.set_description_str("Train epoch {}, Model {}:".format(epoch, model_idx))
                    for batch_idx, batch in enumerate(train_dataloader): # Loop over batches
                        # Run training step for jth model
                        log_params = self.training_step(batch, model_idx)

                        pbar.set_postfix_str("loss: {}".format(log_params['model_{}_loss'.format(model_idx)]))
                        pbar.update(1)

                        if batch_idx % self.log_freq == 0 and self.logger is not None:
                            self.log_metrics(epoch, batch_idx, num_train_batches, log_params)

                self.eval()
                with tqdm(total = num_val_batches, disable = self.disable_bars) as pbar:
                    val_running_counts = None
                    pbar.set_description_str("Validation:".format(epoch))
                    for batch_idx, batch in enumerate(val_dataloader): # Loop over batches
                        # Run training step for jth model
                        log_params = self.validation_step(batch, model_idx)

                        pbar.set_postfix_str("epoch {}, model_idx: {}, loss: {}".format(epoch, model_idx, log_params['model_{}_val_loss'.format(model_idx)]))
                        pbar.update(1)

                        # Add values
                        if(val_running_counts is None):
                            val_running_counts = log_params
                        else:
                            for key, val in val_running_counts.items():
                                val_running_counts[key] += log_params[key]

                    if self.logger is not None:
                        for key, val in val_running_counts.items():
                            val_running_counts[key] /= num_val_batches
                        self.log_metrics(epoch, num_train_batches, num_train_batches, log_params)

            if(epoch % steps_between_model_save == 0):
                self.save("incremental-step-{}".format(epoch, ))

        self.save("final")

    def predict(self, obs, actions, hidden_state = None):
        self.eval()

        feed = self.prepare_feed(obs, actions)

        return self.forward(feed, hidden_state = hidden_state)


    def configure_optimizers(self):
        # Define optimizers
        # self.optimizer is defined in __init__
        # This is the type of optimizer we want to use
        optimizers = [self.optimizer_type(model.parameters()) for model in self.models]

        return optimizers


    def save(self, model_name):
        # Don't save model if no logger configured
        if self.logger is None:
            print("DYNAMICS ENSEMBLE: SKIPPING SAVING SINCE LOGGER IS NOT CONFIGURED")
            return

        print("DYNAMICS ENSEMBLE: Saving model {}".format(model_name))

        # Save model
        self.logger.torch_save(self.state_dict(), ProbabilisticGRUDynamicsEnsemble.model_log_dir, model_name)

    @classmethod
    def load(cls, logger, model_name, gpu):
        # To load the model, we first need to build an instance of this class
        # We want to keep the same config parameters, so we will build it from the pickled config
        # Also, we will load the dimensional parameters of the model from the saved dimensions
        # This allows us to avoid loading a dataloader every time we want to do inference

        print("DYNAMICS ENSEMBLE: Loading dynamics model {}".format(model_name))
        # Get config from pickle first
        config = logger.pickle_load(ProbabilisticGRUDynamicsEnsemble.log_dir, "config.pkl")

        # Next, get pickle containing the state parameters
        state_dim_in, state_dim_out, action_dim, frame_stack, norm_stats = logger.pickle_load(ProbabilisticGRUDynamicsEnsemble.log_dir, "state_params.pkl")
        print("DYNAMICS ENSEMBLE: state_dim_in: {}\tstate_dim_out: {}\taction_dim: {}\tframe_stack: {}".format(
            state_dim_in,
            state_dim_out,
            action_dim,
            frame_stack,
        ))

        # Create a configured dynamics ensemble object
        model = cls(config = config,
                    state_dim_in = state_dim_in,
                    state_dim_out = state_dim_out,
                    action_dim = action_dim,
                    frame_stack = frame_stack,
                    norm_stats = norm_stats,
                    gpu = gpu)

        model.load_state_dict(logger.torch_load(ProbabilisticGRUDynamicsEnsemble.model_log_dir, model_name))

        return model





