from re import S
import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import scipy.spatial

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import gym
from gym.spaces import Box, Discrete, Tuple

# compute reward
from projects.morel_mopo.algorithm.reward import compute_reward
from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils



class BaseFakeEnv(gym.Env):
    def __init__(self, dynamics,
                        config,
                        logger = None):

        # Save logger
        self.logger = logger

        # Save and verify config
        self.config = config
        self.config.verify()



        ################################################
        # Dynamics parameters
        ################################################
        self.dynamics = dynamics

        # device
        self.device_num = self.dynamics.gpu
        self.device = "cuda:{}".format(self.device_num) if torch.cuda.is_available() else "cpu"
        print('Device: ', self.device)
        self.dynamics.to(self.device)

        self.input_dim = self.dynamics.state_dim_in
        self.output_dim = self.dynamics.state_dim_out
        self.action_dim = self.dynamics.action_dim
        print(f'Input dim: {self.input_dim}, Output dim: {self.output_dim}')

        ################################################
        # Creating Action and Observation spaces
        ################################################
        self.action_space = self.config.action_config.action_space
        self.observation_space = self.config.obs_config.observation_space

        self.state = None
        self.vehicle_pose = None
        self.waypoints = None

        ################################################
        # Dataset comes from dynamics
        ################################################
        self.offline_data_module = self.dynamics.data_module
        if(self.offline_data_module is None):
            print("FAKE_ENV: Data module does not have associated dataset. Reset must be called with an initial state input.")

        # Get norm stats and frame_stack from dynamics
        # This will be the same as the norm stats from the original dataset, on which the dynamics model was trained
        self.norm_stats = self.dynamics.normalization_stats
        self.frame_stack = self.dynamics.frame_stack

        ################################################
        # State variables for the fake env
        ################################################
        self.state = feutils.NormalizedTensor(self.norm_stats["obs"]["mean"], self.norm_stats["obs"]["std"], self.device)
        self.past_action = feutils.NormalizedTensor(self.norm_stats["action"]["mean"], self.norm_stats["action"]["std"], self.device)
        self.deltas = feutils.NormalizedTensor(self.norm_stats["delta"]["mean"], self.norm_stats["delta"]["std"], self.device)

        ################################################
        # MOPO hyperparameters
        ################################################
        self.uncertainty_coeff = self.config.uncertainty_coeff
        self.timeout_steps = self.config.timeout_steps

        if(self.logger is not None):
            self.logger.log_scalar("mopo/uncertainty_coeff", self.uncertainty_coeff)
            self.logger.log_scalar("mopo/rollout_length", self.timeout_steps)

    # sample from dataset
    def sample(self):
        return self.offline_data_module.sample_with_waypoints()

    ''' Resets environment. If no input is passed in, sample from dataset '''
    def reset(self, inp=None):

        self.test = False
        # print("FAKE_ENV: Resetting environment...\n")

        if inp is None:
            if(self.offline_data_module is None):
                raise Exception("FAKE_ENV: Cannot sample from dataset since dynamics model does not have associated dataset.")
            ((obs, action, _, _, _, vehicle_pose), waypoints) = self.sample()
            self.state.normalized = obs[:,:2]
            self.past_action.normalized = action

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

        self.waypoints = feutils.filter_waypoints(waypoints).to(self.device)


        if(len(self.waypoints == 2)):
            self.second_last_waypoint = self.waypoints[0]
        else:
            self.second_last_waypoint = None

        if(len(self.waypoints == 1)):
            self.last_waypoint = self.waypoints[0]
        else:
            self.last_waypoint = None



        self.vehicle_pose = vehicle_pose.to(self.device)

        self.steps_elapsed = 0

        self.model_idx = np.random.choice(self.dynamics.n_models)
        # Reset hidden state

        #TODO Return policy features, not dynamics features
        policy_obs, _, _ = self.get_policy_obs(self.state.unnormalized[0])
        return policy_obs.cpu().numpy()

    def calc_disc(self, predictions):
        # Compute the pairwise distances between all predictions
        return scipy.spatial.distance_matrix(predictions, predictions)


    def usad(self, predictions):
        # thres = self.mean + (self.uncertain_threshold * self.beta_max) * (self.std)

        # max_discrep = np.amax(self.calc_disc(predictions)
        # If maximum is greater than threshold, return true
        return np.amax(self.calc_disc(predictions))


    '''
    Updates state vector according to dynamics prediction
    # @params: delta      [Δspeed, Δsteer]
    # @return new state:  [[speed_t+1, steer_t+1], [speed_t, steer_t], speed_t-1, steer_t-1]]
    '''
    def update_next_state(self, past_state, delta_state):
        raise NotImplementedError

    def update_action(self, prev_action, new_action):
        raise NotImplementedError

    def make_prediction(self, past_state, past_action):
        raise NotImplementedError

    def get_policy_obs(self, state):
        angle, \
        dist_to_trajectory, \
        next_waypoints,\
        _, _, \
        remaining_waypoints, \
        self.second_last_waypoint, \
        self.last_waypoint = feutils.process_waypoints(self.waypoints,
                                                        self.vehicle_pose,
                                                        self.device,
                                                        second_last_waypoint = self.second_last_waypoint,
                                                        last_waypoint = self.last_waypoint)

        # convert to tensors
        self.waypoints = torch.FloatTensor(remaining_waypoints)
        dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(self.device)
        angle              = torch.Tensor([angle]).to(self.device)

        return torch.cat([angle, state[1:2] / 10, state[0:1], dist_to_trajectory], dim=0).float().to(self.device), dist_to_trajectory, angle

    '''
    Takes one step according to action
    @ params: new_action
    @ returns next_obs, reward_out, (uncertain or timeout), {"delta" : delta, "uncertain" : 100*uncertain}
    '''
    def step(self, new_action):
        # print('Stepping with action', new_action)
        with torch.no_grad():

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(new_action, torch.Tensor)):
                new_action = torch.FloatTensor(new_action)

            # clamp new action to safe range
            #TODO Switch to using action space to clamp
            new_action = torch.squeeze(torch.clamp(new_action, -1, 1)).to(self.device)

            self.past_action.unnormalized = self.update_action(self.past_action, new_action)

            ############ feed obs, action into dynamics model for prediction ##############

            # input [[speed_t, steer_t, Δtime_t, action_t], [speed_t-1, steer_t-1, Δt-1, action_t-1]]
            # unsqueeze to form batch dimension for dynamics input

            # print(f'Curr vehicle pose: {self.vehicle_pose}')

            # print(f'State: {self.state.unnormalized}, Action: {self.past_action.unnormalized}')

            # Delta: prediction from one randomly selected model
            # [Δx_t+1, Δy_t+1, Δtheta_t+1, Δspeed_t+1, Δsteer_t+1]
            self.deltas.normalized = torch.clone(self.make_prediction(self.state, self.past_action))

            # predicted change in x, y, th
            delta_vehicle_poses = self.deltas.unnormalized[:,:3]

            # change in steer, speed
            delta_state        = self.deltas.unnormalized[self.model_idx,3:5]

            # update vehicle pose
            vehicle_loc_delta = torch.transpose(torch.tensordot(feutils.rot(torch.deg2rad(self.vehicle_pose[2])).to(self.device), delta_vehicle_poses[:,0:2], dims = ([1], [1])), 0, 1)

            # print("FAKE DELTA: {:4f} {:4f} {:4f}".format(vehicle_loc_delta.squeeze()[0].cpu().item(),
            #                                             vehicle_loc_delta.squeeze()[1].cpu().item(),
            #                                             delta_vehicle_poses.squeeze()[2].cpu().item()))
            # print("FAKE TRANSFORMED: {:4f} {:4f} {:4f}".format(delta_vehicle_poses.squeeze()[0].cpu().item(),
            #                                             delta_vehicle_poses.squeeze()[1].cpu().item(),
            #                                             delta_vehicle_poses.squeeze()[2].cpu().item()))

            vehicle_loc = self.vehicle_pose[0:2] + vehicle_loc_delta
            vehicle_rot = self.vehicle_pose[2] + delta_vehicle_poses[:,2]
            vehicle_rot = torch.unsqueeze(vehicle_rot, dim = 1)

            vehicle_poses = torch.cat([vehicle_loc, vehicle_rot], dim = 1)
            self.vehicle_pose = vehicle_poses[self.model_idx]

            if(self.vehicle_pose[2] < -180):
                self.vehicle_pose[2] += 360
            elif(self.vehicle_pose[2] > 180):
                self.vehicle_pose[2] -= 360

            # import ipdb; ipdb.set_trace()

            # update next state
            self.state.unnormalized = self.update_next_state(self.state, delta_state)

            ###################### calculate waypoint features (in global frame)  ##############################

            policy_obs, dist_to_trajectory, angle = self.get_policy_obs(self.state.unnormalized[0])

            # check if at goal
            done = (len(self.waypoints) == 0 or torch.abs(dist_to_trajectory) > 5)

            ################## calc reward with penalty for uncertainty ##############################

            reward_out = compute_reward(self.state.unnormalized[0], dist_to_trajectory, self.config)

            uncertain =  self.usad(self.deltas.normalized.detach().cpu().numpy())
            reward_out[0] = reward_out[0] - uncertain * self.uncertainty_coeff
            reward_out = torch.squeeze(reward_out)

            # advance time
            self.steps_elapsed += 1
            timeout = self.steps_elapsed >= self.timeout_steps

            # log
            if(uncertain and self.logger is not None):
                # self.logger.get_metric("average_halt")
                self.logger.log_hyperparameters({"halts" : 1})
            elif(timeout and self.logger is not None):
                self.logger.log_hyperparameters({"halts" : 0})


            ######################### build policy input ##########################

            # Policy input (unnormalized): dist_to_trajectory, next orientation, speed, steer
            # policy_input = torch.cat([dist_to_trajectory, angle, torch.flatten(self.state.unnormalized[0, :])], dim=0).float().to(self.device)

            info = {
                        "delta" : self.deltas.normalized[self.model_idx].cpu().numpy(),
                        "uncertain" : self.config.uncertainty_coeff * uncertain,
                        "predictions": vehicle_poses.cpu().numpy()
                    }

            res = policy_obs.cpu().numpy(), float(reward_out.item()), timeout or done, info
            return res