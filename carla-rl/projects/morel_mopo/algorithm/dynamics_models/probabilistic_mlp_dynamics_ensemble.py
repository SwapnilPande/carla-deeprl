import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import scipy.spatial
import os
from tqdm import tqdm


class NLLLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(NLLLoss, self).__init__()

    def forward(self, prediction, target):

        # Get neg_log_prob for all batch dim, time step, state dim
        neg_log_prob = - prediction.log_prob(target)
        # Average across state dim
        mean_neg_log_prob = torch.mean(neg_log_prob)

        return mean_neg_log_prob


class LogGaussianLoss(torch.nn.Module):

    def __init__(self) -> None:
        super(NLLLoss, self).__init__()

    def forward(self, x, mu, log_sigma):

        # Get neg_log_prob for all batch dim, time step, state dim
        neg_log_prob = - prediction.log_prob(target)
        # Average across state dim
        mean_neg_log_prob = torch.mean(neg_log_prob)

        return float(-0.5 * np.log(2 * np.pi)) - log_sigma - (x - mu) * 2 / (2 * torch.exp(log_sigma) * 2)



class ProbabilisticDynamicsMLP(nn.Module):
    def __init__(self,
                state_dim_in,
                state_dim_out,
                action_dim,
                frame_stack,
                predict_reward = False,
                n_neurons = 200,
                n_hidden_layers = 4,
                n_head_layers = 1,
                drop_prob = 0,
                activation = nn.ReLU):
        super(ProbabilisticDynamicsMLP, self).__init__()


        #print(state_dim_in,
        #        state_dim_out,
        #        action_dim,
        #        frame_stack)
        # Validate inputs
        assert state_dim_in > 0
        assert state_dim_out > 0
        assert n_neurons > 0
        assert drop_prob >= 0 and drop_prob < 1

        # Store configuration parameters

        # Input is the input state plus actions state
        self.input_dim = int(frame_stack*(state_dim_in + action_dim))
        # Output is the output state dim + reward if True
        self.output_dim = (state_dim_out + int(predict_reward))
        self.state_dim_out = state_dim_out
        self.predict_reward = predict_reward

        self.n_neurons = n_neurons

        # layer list
        layer_list = []
        # add input list and activation
        layer_list.append(nn.Linear(self.input_dim, n_neurons))
        layer_list.append(activation())

        # add hidden layers
        for _ in range(n_hidden_layers):
            # Add Linear layer
            layer_list.append(nn.Linear(n_neurons, n_neurons))

            # Add activation
            layer_list.append(activation())

            # add dropout if enabled
            if(drop_prob != 0):
                layer_list.append(nn.Dropout(p = drop_prob))

        self.model = nn.ModuleList(layer_list)

        # add last head layer to match the output size
        self.mean_state_head = nn.Linear(n_neurons, self.state_dim_out)
        self.std_state_head = nn.Linear(n_neurons, self.state_dim_out)
        self.std_act = torch.exp

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

    # one single prediction in the form of gaussian
    # the output is [[meanhat1, meanhat2...], [varhat1, varhat2...], ([rewardhat1, rewardhat2...])]
    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        mean = self.mean_state_head(x)
        std = self.std_state_head(x)
        std = self.std_act(std)

        reward_out = None
        if self.reward_head is not None:
            reward_out = self.reward_head[0](x)
            for layer in self.reward_head[1:]:
                reward_out = layer(reward_out)
            output = torch.cat([output, reward_out], dim = 1)

        return mean, std, reward_out


class ProbabilisticMLPDynamicsEnsemble(nn.Module):
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
                    log_freq = 1200,
                    disable_bars = True):
        super(ProbabilisticMLPDynamicsEnsemble, self).__init__()

        self.config = config

        # Save the logger
        self.logger = logger

        self.disable_bars = disable_bars

        # Save config to load in the future
        if logger is not None:
            self.logger.pickle_save(self.config, ProbabilisticMLPDynamicsEnsemble.log_dir, "config.pkl")


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
            self.logger.pickle_save(state_params, ProbabilisticMLPDynamicsEnsemble.log_dir, "state_params.pkl")

        # Model parameters
        self.n_models = config.n_models
        self.network_cfg = config.network_cfg

        self.lr = config.lr

        self.loss = NLLLoss()

        # Save optimizer object
        self.optimizer_type = config.optimizer_type
        self.optimizer_args = config.optimizer_args

        # Setup GPU
        self.gpu = self.config.gpu
        if self.gpu == -1:
            self.device = "cpu"
        else:
            self.device = "cuda:{}".format(self.gpu)

        # Create n_models models
        self.models = nn.ModuleList()
        for i in range(self.n_models):
            self.models.append(ProbabilisticDynamicsMLP(
                state_dim_in = self.state_dim_in,
                state_dim_out = self.state_dim_out,
                action_dim = self.action_dim,
                frame_stack = self.frame_stack,
                predict_reward = self.network_cfg.predict_reward,
                n_neurons = self.network_cfg.n_neurons,
                n_hidden_layers = self.network_cfg.n_hidden_layers,
                n_head_layers = self.network_cfg.n_head_layers,
                drop_prob = 0,
                activation = self.network_cfg.activation
            ))
            self.models[-1].to(self.device)

    def forward(self, x, model_idx=None, return_dist = False):
        # Create a list of the models to iterate over
        if model_idx is None:
            model_indices = list(range(self.n_models))
        else:
            model_indices = [model_idx]


        state_predictions = []
        reward_predictions = []
        # predict for all models
        for model_i in model_indices:
            # Get output from model
            mean, std, reward = self.models[model_i](x)

            print(torch.linalg.norm(std))

            # Build diagonal gaussian model
            try:
                dist = torch.distributions.normal.Normal(
                                loc = mean,
                                scale = std
                            )
            except:
                import ipdb; ipdb.set_trace()

            # If not return distribtuon, then sample from the distribution
            if(not return_dist):
                output = dist.sample()
            # Return (dist, reward)
            else:
                output = dist

            state_predictions.append(output)
            reward_predictions.append(reward)

        return state_predictions, reward_predictions

    def prepare_feed(self, state_in, actions):
        # Convert to cuda first
        state_in = state_in.to(self.device)
        actions = actions.to(self.device)

        # state_in is [batch_size, frame_stack, state_dim]
        # action is [batch_size, frame_stack, action_dim]
        # feed tensor is [batch_size, frame_stack * (state_dim + action_dim)]
        feed_tensor = torch.reshape(
                            torch.cat([state_in, actions], dim = -1),
                                (-1, self.frame_stack * (self.state_dim_in + self.action_dim)
                            )
                      )

        return feed_tensor

    def prepare_target(self, delta, reward):
        delta = delta.to(self.device)
        reward = reward.to(self.device)

        if(self.network_cfg.predict_reward):
            return torch.cat([delta,reward], dim = -1)

        return delta

    def prepare_batch(self, state_in, actions, delta, reward):
        # Input should be obs and actions concatenated, and flattened
        feed_tensor = self.prepare_feed(state_in, actions)

        target_tensor = self.prepare_target(delta, reward)

        # Else, just return delta
        return feed_tensor, target_tensor

    def training_step(self, batch, model_idx):
        # Zero Optimizer gradients
        self.optimizers[model_idx].zero_grad()

        # Split batch into componenets
        obs, actions, rewards, delta, done, vehicle_pose = batch

        # Combine tensors and reshape batch to flat inputs
        feed, target = self.prepare_batch(obs, actions, delta, rewards)

        # Make prediction with selected model
        y_hat, reward = self.forward(feed, model_idx = model_idx, return_dist=True)
        y_hat = y_hat[0]

        loss = self.loss(y_hat, target)
        if(torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss))):
            import ipdb; ipdb.set_trace()

        # Backpropagate
        loss.backward()

        # Take optimizer step
        self.optimizers[model_idx].step()

        # Return metrics to log
        return {"model_{}_loss".format(model_idx) : loss}


    def validation_step(self, batch, model_idx = 0):
        with torch.no_grad():
            # Split batch into componenets
            obs, actions, rewards, delta, done, vehicle_pose = batch

            # Combine tensors and reshape batch to flat inputs
            feed, target = self.prepare_batch(obs, actions, delta, rewards)

            # Predictions by each model

            y_hat, reward = self.forward(feed, model_idx = model_idx, return_dist=True)
            y_hat = y_hat[0]

            # Compute loss

            loss = self.loss(y_hat, target)
            #chaged: loss function
            #mse_loss = self.mse_loss(y_hat, target)

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
                "dyn_ens/n_neurons" : self.network_cfg.n_neurons,
                "dyn_ens/n_hidden_layers" : self.network_cfg.n_hidden_layers,
                "dyn_ens/n_head_layers" : self.network_cfg.n_head_layers,
                "dyn_ens/frame_stack" : self.frame_stack,
                "dyn_ens/drop_prob" : 0,
                "dyn_ens/activation" : str(self.network_cfg.activation)

            })


        # Configure the optimizers
        self.optimizers = self.configure_optimizers()

        # Only run validation if val dataloader has elements
        if(len(val_dataloader) <= 0):
            print("WARNING: Skipping validation since validation dataloader has 0 elements.")


        steps_between_model_save = epochs // n_incremental_models
        if(steps_between_model_save == 0):
            steps_between_model_save = 1

        for epoch in range(epochs): # Loop over epochs
            for model_idx in range(self.n_models): # Loop over models

                self.train()
                with tqdm(total = num_train_batches) as pbar:
                    pbar.set_description_str("Train epoch {}, Model {}:".format(epoch, model_idx))
                    for batch_idx, batch in enumerate(train_dataloader): # Loop over batches
                        # Run training step for jth model
                        log_params = self.training_step(batch, model_idx)

                        pbar.set_postfix_str("loss: {}".format(log_params['model_{}_loss'.format(model_idx)]))
                        pbar.update(1)

                        if batch_idx % self.log_freq == 0 and self.logger is not None:
                            self.log_metrics(epoch, batch_idx, num_train_batches, log_params)

                self.eval()
                with tqdm(total = num_val_batches) as pbar:
                    pbar.set_description_str("Validation:".format(epoch))
                    for batch_idx, batch in enumerate(val_dataloader): # Loop over batches
                        # Run training step for jth model
                        log_params = self.validation_step(batch, model_idx)

                        pbar.set_postfix_str("epoch {}, model_idx: {}, loss: {}".format(epoch, model_idx, log_params['model_{}_val_loss'.format(model_idx)]))
                        pbar.update(1)

                        if batch_idx % self.log_freq == 0 and self.logger is not None:
                            self.log_metrics(epoch, batch_idx, num_val_batches, log_params)

            if(epoch % steps_between_model_save == 0):
                self.save("incremental-step-{}".format(epoch, ))

        self.save("final")


    def configure_optimizers(self):
        # Define optimizers
        # self.optimizer is defined in __init__
        # This is the type of optimizer we want to use
        optimizers = [self.optimizer_type(model.parameters()) for model in self.models]

        return optimizers

    def predict(self, obs, actions):
        self.eval()

        feed = self.prepare_feed(obs, actions)

        return self.forward(feed)


    def save(self, model_name):
        # Don't save model if no logger configured
        if self.logger is None:
            print("DYNAMICS ENSEMBLE: SKIPPING SAVING SINCE LOGGER IS NOT CONFIGURED")
            return

        print("DYNAMICS ENSEMBLE: Saving model {}".format(model_name))

        # Save model
        self.logger.torch_save(self.state_dict(), ProbabilisticMLPDynamicsEnsemble.model_log_dir, model_name)

    @classmethod
    def load(cls, logger, model_name, gpu):
        # To load the model, we first need to build an instance of this class
        # We want to keep the same config parameters, so we will build it from the pickled config
        # Also, we will load the dimensional parameters of the model from the saved dimensions
        # This allows us to avoid loading a dataloader every time we want to do inference

        print("DYNAMICS ENSEMBLE: Loading dynamics model {}".format(model_name))
        # Get config from pickle first
        config = logger.pickle_load(ProbabilisticMLPDynamicsEnsemble.log_dir, "config.pkl")

        # Next, get pickle containing the state parameters
        state_dim_in, state_dim_out, action_dim, frame_stack, norm_stats = logger.pickle_load(ProbabilisticMLPDynamicsEnsemble.log_dir, "state_params.pkl")
        print("DYNAMICS ENSEMBLE: state_dim_in: {}\tstate_dim_out: {}\taction_dim: {}\tframe_stack: {}".format(
            state_dim_in,
            state_dim_out,
            action_dim,
            frame_stack,
        ))

        # Create a configured dynamics ensemble object
        return cls(config = config,
                    state_dim_in = state_dim_in,
                    state_dim_out = state_dim_out,
                    action_dim = action_dim,
                    frame_stack = frame_stack,
                    norm_stats = norm_stats,
                    gpu = gpu)
