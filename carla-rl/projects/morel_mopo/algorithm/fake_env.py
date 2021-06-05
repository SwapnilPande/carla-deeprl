import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import scipy.spatial

# torch imports
import torch
print("REDUCED STEER REWARD")

# def unnormalize_deltas(deltas):
#     # assert len(deltas.shape) == 1
#     deltas = deltas*(torch.tensor([0.0073, 0.0241, 0.0076, 0.0138]).to("cuda:2")) + torch.tensor([-4.0775e-06, -4.5966e-05,  1.6651e-04,  2.2809e-06]).to("cuda:2")

#     return deltas

# def unnormalize_mlp(mlp_features):
#     # assert len(mlp_features.shape) == 1
#     # mlp_features[0] = mlp_features[0]*(7*10**2)
#     # mlp_features[1] = mlp_features[1]*(3*10**2)
#     # mlp_features[2] = mlp_features[2]*(2)
#     # mlp_features[2] = mlp_features[2]*(2*10**2)

#     mlp_features = mlp_features[:,0:4]*torch.tensor([0.0327, 0.0781, 0.1413, 0.0785]).to("cuda:2") + torch.tensor([ 0.0003, -0.0039,  0.1515,  0.0010]).to("cuda:2")

#     return mlp_features

# def normalize_mlp(mlp_features):

#     mlp_features = (mlp_features[:,0:4] - torch.tensor([ 0.0003, -0.0039,  0.1515,  0.0010]).to("cuda:2"))/torch.tensor([0.0327, 0.0781, 0.1413, 0.0785]).to("cuda:2")

#     return mlp_features

# def unnormalize_deltas(deltas):
#     # assert len(deltas.shape) == 1
#     # deltas = deltas*(torch.tensor([0.0073, 0.0241, 0.0076, 0.0138]).to("cuda:2")) + torch.tensor([-4.0775e-06, -4.5966e-05,  1.6651e-04,  2.2809e-06]).to("cuda:2")

#     return deltas

# def unnormalize_mlp(mlp_features):
#     # assert len(mlp_features.shape) == 1
#     # mlp_features[0] = mlp_features[0]*(7*10**2)
#     # mlp_features[1] = mlp_features[1]*(3*10**2)
#     # mlp_features[2] = mlp_features[2]*(2)
#     # mlp_features[2] = mlp_features[2]*(2*10**2)

#     # mlp_features = mlp_features[:,0:4]*torch.tensor([0.0327, 0.0781, 0.1413, 0.0785]).to("cuda:2") + torch.tensor([ 0.0003, -0.0039,  0.1515,  0.0010]).to("cuda:2")

#     return mlp_features

# def normalize_mlp(mlp_features):

#     # mlp_features = (mlp_features[:,0:4] - torch.tensor([ 0.0003, -0.0039,  0.1515,  0.0010]).to("cuda:2"))/torch.tensor([0.0327, 0.0781, 0.1413, 0.0785]).to("cuda:2")

#     return mlp_features


def unnormalize_state(obs, device):
    return torch.tensor([0.9105, 3.3075, 0.1100, 0.1323,1.0]).to(device)*obs + torch.tensor([ 0.2295,  3.5232,  0.0196, -0.0883,0]).to(device)

def normalize_state(obs, device):
    return (obs - torch.tensor([ 0.2295,  3.5232,  0.0196, -0.0883,0.0]).to(device))/torch.tensor([0.9105, 3.3075, 0.1100, 0.1323,1.0]).to(device)

def unnormalize_delta(delta, device):
    return torch.tensor([0.2927, 0.2108, 0.0491, 0.0334,1.0]).to(device)*delta

def normalize_delta(delta, device):
    return delta/torch.tensor([0.2927, 0.2108, 0.0491, 0.0334,1.0]).to(device)

def calc_next_state(obs, delta, device):
    return normalize_state(unnormalize_state(obs, device) + unnormalize_delta(delta, device), device)



class FakeEnv:
    def __init__(self, dynamics,
                        logger = None,
                        uncertainty_threshold = 0.5,
                        uncertain_penalty = -100,
                        timeout_steps = 1,
                        uncertainty_params = [0.0045574815320799725, 1.9688976602303934e-05, 0.2866033549975823]):


        # MOReL hyperparameters
        self.uncertain_threshold = uncertainty_threshold
        self.uncertain_penalty = uncertain_penalty
        self.timeout_steps = timeout_steps


        # Get dynamics model parameters
        self.dynamics = dynamics
        self.input_dim, self.output_dim = self.dynamics.get_input_output_dim()
        self.device_num = self.dynamics.get_gpu()
        self.device = "cuda:{}".format(self.device_num)

        self.logger = logger
        self.state = None

        # Setup dataset
        self.offline_data_module = self.dynamics.get_data_module()
        self.dataloader = self.offline_data_module.train_dataloader(weighted = False, batch_size_override = 1)
        # Move dynamics to correct device
        # We only have to do this because lightning moves the device back to CPU
        self.dynamics.to(self.device)

        self.data_iter = iter(self.dataloader)

        # self.calc_usad_params()

        # self.mean = 0
        # self.var = 1
        # self.std = np.sqrt(self.var)
        # self.maximum = 2
        # self.beta_max = 2

    def sample(self):
        # print("_--------------------------")
        # print(len(self.dataloader))
        # print(type(len(self.dataloader)))
        # print("----------------------------")

        # sample_num = torch.randint(len(self.dataloader), size = (1,))
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def calculate_reward(self, cur_state, next_state):

        cur_state_unnorm = unnormalize_state(cur_state, device = self.device)
        next_state_unnorm = unnormalize_state(next_state, device = self.device)

        speed = next_state_unnorm[1]*15

        steer_reward = torch.abs(cur_state_unnorm[2] - cur_state_unnorm[2])
        dist_to_trajectory = next_state_unnorm[0]*10

        off_route = torch.abs(dist_to_trajectory) > 10
        collision = torch.abs(next_state_unnorm[-1]) < 0.2

        return torch.unsqueeze(speed - 1.2*torch.abs(dist_to_trajectory) - (off_route * 50.) - 1.3*steer_reward - (collision * 50), dim = 0)

    def reset(self, obs = None, action = None):
        if(obs is None):
            obs, action, _, _, _ = self.sample()
            self.state = torch.clone(obs)
            self.state =  self.state.to(self.device)
            self.past_action = action[:,2:4].to(self.device)
        else:
            obs = torch.tensor(obs).to(self.device)
            self.state = torch.clone(obs)
            # obs = normalize_mlp(obs)
            action = torch.tensor(action).to(self.device)
            self.past_action = action[:,2:4]


        self.steps_elapsed = 0

        return self.state[:,[0,2,4,6,16]]

    def calc_usad_params(self):
        # Mask to select upper triangle of the distance matrix
        # Does not include diagonal
        mask = np.triu_indices(self.dynamics.get_n_models(), 1)

        print("Calculating disagreement statistics")

        uncertainty_sum = 0.0
        uncertainty_sum_sq = 0.0
        uncertainty_maximum = 0.0
        n = 0

        for i, batch in enumerate(tqdm(self.dataloader)):
            obs, action, _, _, _ = batch
            x = torch.cat([obs, action], dim = 1).float().to(self.device)

            predictions = self.dynamics.predict(x).cpu().numpy()
            predictions = np.swapaxes(predictions, 0, 1)
            # predictions = predictions[:,:,:-1]


            # Generate distance matrix for predictions
            for prediction in predictions:
                discs = self.calc_disc(prediction)
                discs_flat = discs[mask]

                # Get maximum
                temp_max = np.amax(discs_flat)
                uncertainty_maximum = np.maximum(temp_max, uncertainty_maximum)

                uncertainty_sum += np.sum(discs_flat)
                uncertainty_sum_sq += np.sum(np.square(discs_flat))
                n += len(discs_flat)

        self.mean = uncertainty_sum/n
        self.var = uncertainty_sum_sq/n - self.mean*self.mean
        self.std = np.sqrt(self.var)
        self.maximum = uncertainty_maximum

        self.beta_max = (self.maximum - self.mean)/self.std

        print("mean: {}".format(self.mean))
        print("var: {}".format(self.var))
        print("maximum: {}".format(self.maximum))

    def calc_disc(self, predictions):
        # Compute the pairwise distances between all predictions
        return scipy.spatial.distance_matrix(predictions, predictions)


    def usad(self, predictions):
        # thres = self.mean + (self.uncertain_threshold * self.beta_max) * (self.std)

        # max_discrep = np.amax(self.calc_disc(predictions)
        # If maximum is greater than threshold, return true
        return np.amax(self.calc_disc(predictions))


    def step(self, action_unnormalized, obs = None):
        action = torch.unsqueeze(action_unnormalized, dim = 0)

        # Clamp actions to safe range
        action = torch.clamp(action, -12, 12).to(self.device)

        action = torch.cat([action, self.past_action], dim = 1)

        # Next, save observation if passed
        # Only used to intialize model to certain state
        if obs is not None:
            if(not torch.is_tensor(obs)):
                obs = torch.unsqueeze(torch.tensor(obs).float().to(self.device), dim = 0)
                self.state = torch.clone(obs)
                # obs = normalize_mlp(obs)
            else:
                obs = obs.to(self.device)
                self.state = torch.clone(obs)
                # obs = normalize_mlp(obs)
        else:
            obs = torch.clone(self.state)
        #     obs = normalize_mlp(self.state)

        # Feed predictions through dynamics model
        # print(self.state.shape)
        predictions = torch.squeeze(self.dynamics.predict(torch.cat([obs, action],dim = 1).float()))
        # predictions = torch.squeeze(self.dynamics.predict(obs.float()))

        # Randomly sample a model and split output
        model_idx = np.random.choice(self.dynamics.n_models)
        # delta = unnormalize_deltas(predictions[model_idx])
        delta = torch.clone(predictions[model_idx])
        #reward = #predictions[model_idx, -1:]

        # Calculate next state
        prev_state = torch.clone(self.state)
        self.state = calc_next_state(self.state[0,[0,2,4,6,16]], delta, self.device)



        next_obs = torch.clone(self.state)

        # next_obs = normalize_mlp(self.state)
        obs = next_obs

        # Next, calculate reward
        reward_out = self.calculate_reward(prev_state[0,[0,2,4,6,16]], self.state)

        # Check if uncertain
        # If uncertain, apply uncertainty penalty
        uncertain = self.usad(predictions.cpu().numpy())
        # if(uncertain):
        reward_out[0] = reward_out[0] - uncertain * 150
        reward_out = torch.squeeze(reward_out)

        self.steps_elapsed += 1

        timeout = self.steps_elapsed >= self.timeout_steps

        if(uncertain and self.logger is not None):
            # self.logger.get_metric("average_halt")
            self.logger.log_metrics({"halts" : 1})
        elif(timeout and self.logger is not None):
            self.logger.log_metrics({"halts" : 0})

        return self.state, reward_out, (uncertain or timeout), {"delta" : delta, "uncertain" : 100*uncertain}
