import torch
import numpy as np

# compute reward
from projects.morel_mopo.algorithm.fake_envs.base_fake_env import BaseFakeEnv
from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils


class RNNFakeEnv(BaseFakeEnv):
    ''' Resets environment. If no input is passed in, sample from dataset '''
    def reset(self, inp=None):
        self.test = False
        # print("FAKE_ENV: Resetting environment...\n")

        if inp is None:
            if(self.offline_data_module is None):
                raise Exception("FAKE_ENV: Cannot sample from dataset since dynamics model does not have associated dataset.")

            max_init_feed_length = 5

            ((obs, action, _, _, _, vehicle_pose, mask), waypoints) = self.sample()

            max_init_feed_length = int(np.minimum(torch.sum(mask).cpu().item() - 1, max_init_feed_length))

            rnn_init_state = obs[0:max_init_feed_length,:2]
            rnn_init_action = action[0:max_init_feed_length]

            self.state.normalized = obs[max_init_feed_length:max_init_feed_length+1,:2]
            self.past_action.normalized = action[max_init_feed_length-1:max_init_feed_length]
            vehicle_pose = vehicle_pose[max_init_feed_length]

        else:
            (obs, action, vehicle_pose, waypoints) = inp

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(obs, torch.Tensor)):
                obs = torch.FloatTensor(obs)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(action, torch.Tensor)):
                action = torch.FloatTensor(action)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(vehicle_pose, torch.Tensor)):
                vehicle_pose = torch.FloatTensor(vehicle_pose)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(waypoints, torch.Tensor)):
                waypoints = torch.FloatTensor(waypoints)


            # state only includes speed, steer
            self.state.unnormalized =  obs[:,:2]
            self.past_action.unnormalized = action

            rnn_init_state = torch.clone(self.state.normalized)
            rnn_init_action = torch.clone(self.past_action.normalized)

        self.waypoints = feutils.filter_waypoints(waypoints).to(self.device)

        if(len(self.waypoints) == 2):
            self.second_last_waypoint = self.waypoints[0]
        else:
            self.second_last_waypoint = None

        if(len(self.waypoints) == 1):
            self.last_waypoint = self.waypoints[0]
        else:
            self.last_waypoint = None

        self.vehicle_pose = vehicle_pose.to(self.device)

        self.steps_elapsed = 0

        self.model_idx = np.random.choice(self.dynamics.n_models)
        # Reset hidden state

        #TODO Return policy features, not dynamics features
        policy_obs, _, _ = self.get_policy_obs(self.state.unnormalized[0])

        # Reset hidden state
        self.hidden_state = [torch.zeros(size = (1, 1, self.dynamics.network_cfg.gru_hidden_dim)).to(self.device) for _ in range(self.dynamics.n_models)]

        # Get predictions across all models
        _, self.hidden_state = self.dynamics.predict(rnn_init_state.unsqueeze(0), rnn_init_action.unsqueeze(0), hidden_state = self.hidden_state)

        return policy_obs.cpu().numpy()

    '''
    Updates state vector according to dynamics prediction
    # @params: delta      [Δspeed, Δsteer]
    # @return new state:  [[speed_t+1, steer_t+1], [speed_t, steer_t], speed_t-1, steer_t-1]]
    '''
    def update_next_state(self, prev_state, delta_state):
        # Return unnormalized next state
        return prev_state.unnormalized[-1:, :] + delta_state

    def update_action(self, prev_action, new_action):
        # Return unnormalized new action
        return new_action.unsqueeze(0)

    def make_prediction(self, past_state, past_action):
        # Get predictions across all models
        predictions, self.hidden_state = self.dynamics.predict(
            past_state.normalized[-1].unsqueeze(0).unsqueeze(0),
            past_action.normalized[-1].unsqueeze(0).unsqueeze(0),
            hidden_state = self.hidden_state)

        # Extract the state prediction, ignore the reward
        state_predictions = torch.stack(predictions[0])

        state_predictions = state_predictions.squeeze(dim = 1).squeeze(dim = 1)

        return state_predictions


    # def get_policy_obs(self):
    #     angle, dist_to_trajectory, next_waypoints, _, _, remaining_waypoints = process_waypoints(self.waypoints, self.vehicle_pose, self.device)

    #     # convert to tensors
    #     self.waypoints = torch.FloatTensor(remaining_waypoints)
    #     dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(self.device)
    #     angle              = torch.Tensor([angle]).to(self.device)

    #     return torch.cat([dist_to_trajectory, angle, torch.flatten(self.state.unnormalized[-1, :])], dim=0).float().to(self.device), dist_to_trajectory, angle
