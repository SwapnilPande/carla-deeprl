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

'''
Calculates L2 distance between waypoint, vehicle
@param waypoint:     torch.Tensor([x,y])
        vehicle_pose: torch.Tensor([x, y, yaw])
'''
def distance_vehicle(waypoint, vehicle_pose, device):
    waypoint = torch.FloatTensor(waypoint).to(device)
    vehicle_loc = vehicle_pose[:2].to(device) # get x, y
    dist = torch.dist(waypoint, vehicle_loc).item()
    return dist


'''
Calculates dot product, angle between vehicle, waypoint
@param waypoint:     [x, y]
       vehicle_pose: torch.Tensor([x, y, yaw])
'''
def get_dot_product_and_angle(vehicle_pose, waypoint, device):
    waypoint = torch.FloatTensor(waypoint).to(device)

    v_begin         = vehicle_pose[:2]
    vehicle_yaw     = vehicle_pose[2]
    v_end = v_begin + torch.Tensor([torch.cos(torch.deg2rad(vehicle_yaw)), \
                                    torch.sin(torch.deg2rad(vehicle_yaw))]).to(device)

    # vehicle vector: direction vehicle is pointing in global coordinates
    v_vec = torch.sub(v_end, v_begin)
    # vector from vehicle's position to next waypoint
    w_vec = torch.sub(waypoint, v_begin)
    # steering error: angle between vehicle vector and vector pointing from vehicle loc to
    # waypoint
    dot   = torch.dot(w_vec, v_vec)
    angle = torch.acos(torch.clip(dot /
                                (torch.linalg.norm(w_vec) * torch.linalg.norm(v_vec)), -1.0, 1.0))


    assert(torch.isclose(torch.cos(angle), torch.clip(torch.dot(w_vec, v_vec) / \
        (torch.linalg.norm(w_vec) * torch.linalg.norm(v_vec)), -1.0, 1.0), atol=1e-3))

    # make vectors 3D for cross product
    v_vec_3d = torch.hstack((v_vec, torch.Tensor([0]).to(device)))
    w_vec_3d = torch.hstack((w_vec, torch.Tensor([0]).to(device)))

    _cross = torch.cross(v_vec_3d, w_vec_3d)

    # if negative steer, turn left
    if _cross[2] < 0:
        angle *= -1.0

    # assert cross product a x b = |a||b|sin(angle)
    # assert(torch.isclose(_cross[2], torch.norm(v_vec_3d) * torch.norm(w_vec_3d) * torch.sin(angle), atol=1e-2))

    return dot, angle, w_vec

'''
Gets distance of vehicle to a line formed by two waypoints
@param waypoint1:     [x,y]
       waypoint2:     [x,y]
       vehicle_pose:  torch.Tensor([x,y, yaw])
'''
def vehicle_to_line_distance(vehicle_pose, waypoint1, waypoint2, device):
    waypoint1 = torch.FloatTensor(waypoint1).to(device)
    waypoint2 = torch.FloatTensor(waypoint2).to(device)

    vehicle_loc   = vehicle_pose[:2] # x, y coords

    a_vec = torch.sub(waypoint2, waypoint1)   # forms line between two waypoints
    b_vec = torch.sub(vehicle_loc, waypoint1) # line from vehicle to first waypoint

    # make 3d for cross product
    a_vec_3d = torch.hstack((a_vec, torch.Tensor([0]).to(device)))
    b_vec_3d = torch.hstack((b_vec, torch.Tensor([0]).to(device)))

    dist_vec = torch.cross(a_vec_3d, b_vec_3d) / torch.linalg.norm(a_vec_3d)
    return abs(dist_vec[2]) # dist

'''
Calculate dist to trajectory, angle
@param waypoints:    [torch.Tensor([wp1_x,wp1_y]), torch.Tensor([wp2_x,wp2_y)......]
       vehicle pose: torch.Tensor([x,y,yaw])
@returns dist_to_trajectory, angle, ...
'''
def process_waypoints(waypoints, vehicle_pose, device):
    vehicle_pose.to(device)
    waypoints = waypoints.tolist()

    next_waypoints_angles = []
    next_waypoints_vectors = []
    next_waypoints = []
    num_next_waypoints = 5
    last_waypoint, second_last_waypoint = None, None

    # closest wp to car
    min_dist_index = -1
    min_dist = np.inf

    for i, waypoint in enumerate(waypoints):
        # find wp that yields min dist between itself and car
        dist_i = distance_vehicle(waypoint, vehicle_pose, device)
        if dist_i < min_dist:
            min_dist_index = i
            min_dist = dist_i

    wp_len = len(waypoints)
    if min_dist_index >= 0:
        # pop waypoints up until the one with min distance to vehicle
        for i in range(min_dist_index + 1):
            waypoint = waypoints.pop(0)
            # set last, second-to-last waypoints
            if i == wp_len - 1:
                last_waypoint = waypoint
            elif i == wp_len - 2:
                second_last_waypoint = waypoint



    # only keep next N waypoints
    for i, waypoint in enumerate(waypoints[:num_next_waypoints]):
        # dist to waypoint
        dot, angle, w_vec = get_dot_product_and_angle(vehicle_pose, waypoint, device)

        if len(next_waypoints_angles) == 0:
            next_waypoints_angles = [angle]
            next_waypoints = [waypoint]
            next_waypoints_vectors = [w_vec]
        else:
            next_waypoints_angles.append(angle)
            next_waypoints.append(waypoint)
            next_waypoints_vectors.append(w_vec)


    # get mean of all angles to figure out which direction to turn
    if len(next_waypoints_angles) > 0:
        angle = torch.mean(torch.FloatTensor(next_waypoints_angles))
    else:
        print("No next waypoint found!")
        angle = 0

    if len(next_waypoints) > 1:
        # get dist from vehicle to a line formed by the next two wps
        dist_to_trajectory = vehicle_to_line_distance(
                                vehicle_pose,
                                next_waypoints[0],
                                next_waypoints[1],
                                device)

    # if only one next waypoint, use it and second to last
    elif len(next_waypoints) > 0:
        dist_to_trajectory = vehicle_to_line_distance(
                                vehicle_pose,
                                second_last_waypoint,
                                next_waypoints[0],
                                device)

    else:

        if second_last_waypoint and last_waypoint:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    second_last_waypoint,
                                    last_waypoint,
                                    device)

        else:
            dist_to_trajectory = 0

    return angle, dist_to_trajectory, next_waypoints, next_waypoints_angles, next_waypoints_vectors



class FakeEnv(gym.Env):
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
        self.observation_space = self.config.obs_config.obs_space

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
        self.norm_stats = self.offline_data_module.normalization_stats
        self.frame_stack = self.dynamics.frame_stack

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
        print("FAKE_ENV: Resetting environment...\n")

        if inp is None:
            if(self.offline_data_module is None):
                raise Exception("FAKE_ENV: Cannot sample from dataset since dynamics model does not have associated dataset.")
            ((obs, action, _, _, _, vehicle_pose), waypoints) = self.sample()
        else:
            (obs, action, vehicle_pose, waypoints) = inp

        self.obs = torch.squeeze(obs).to(self.device)
        self.past_action = torch.squeeze(action).to(self.device)
        self.waypoints = torch.squeeze(waypoints).to(self.device)
        self.vehicle_pose = torch.squeeze(vehicle_pose).to(self.device)
        # state only includes speed, steer
        self.state =  self.obs[:, :2].to(self.device)

        print('obs', self.obs.shape)
        print('action', self.past_action.shape)
        print('waypoints', self.waypoints.shape)
        print('vehicle pose', self.vehicle_pose.shape)
        print('state', self.state.shape)

        self.steps_elapsed = 0

        # dist to traj, orientation, self.state
        wp_features = np.array([np.inf, 90])
        obs = np.hstack((wp_features, torch.flatten(self.state[0, :]).cpu().detach().numpy()))
        return obs


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
    def update_next_state(self, delta_state):
        # calculate newest state
        newest_state = self.state[0, :] + delta_state
        # insert newest state at front
        self.state = torch.cat([newest_state.unsqueeze(0), self.state], dim=0)
        # delete oldest state
        self.state = self.state[:-1, :]
        return self.state

    '''
    Takes one step according to action
    @ params: new_action
    @ returns next_obs, reward_out, (uncertain or timeout), {"delta" : delta, "uncertain" : 100*uncertain}
    '''
    def step(self, new_action, obs = None):
        new_action = torch.tensor(new_action).squeeze().to(self.device)
        # clamp new action to safe range
        new_action = torch.clamp(new_action, -1, 1).to(self.device)
        # insert new action at front, delete oldest action
        action = torch.cat([new_action.unsqueeze(0), self.past_action[:-1, :]])

        # if obs not passed in
        if not obs:
            obs = self.obs

        ############ feed obs, action into dynamics model for prediction ##############

        # input [[speed_t, steer_t, Δtime_t, action_t], [speed_t-1, steer_t-1, Δt-1, action_t-1]]
        # unsqueeze to form batch dimension for dynamics input
        dynamics_input = torch.cat([obs, action.reshape(-1,2)], dim = 1).unsqueeze(0).float()

        # Get predictions across all models
        all_predictions = torch.stack(self.dynamics.forward(torch.flatten(dynamics_input)))

        # Delta: prediction from one randomly selected model
        # [Δx_t+1, Δy_t+1, Δtheta_t+1, Δspeed_t+1, Δsteer_t+1]
        model_idx = np.random.choice(self.dynamics.n_models)
        delta = all_predictions[model_idx]
        delta = self.unnormalize_delta(delta, self.device)
        # predicted change in x, y, th
        delta_vehicle_pose = delta[:3]
        # change in speed, steer
        delta_state        = delta[3:5]

        # update vehicle pose
        self.vehicle_pose = self.vehicle_pose + delta_vehicle_pose
        # update next state
        self.state = self.update_next_state(delta_state)


        ###################### calculate waypoint features (in global frame)  ##############################

        self.waypoints = self.waypoints[:, :2] # get x,y coords for each waypoint
        angle, dist_to_trajectory, _, _, _ = process_waypoints(self.waypoints, self.vehicle_pose, self.device)

        ################## calc reward with penalty for uncertainty ##############################

        reward_out = compute_reward(self.state, dist_to_trajectory, self.config)

        uncertain =  self.usad(all_predictions.detach().cpu().numpy())
        reward_out[0] = reward_out[0] - uncertain * self.config.uncertainty_coeff
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
        dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(self.device)
        angle              = torch.Tensor([angle]).to(self.device)


        # combine most recent state with waypoint features to form policy input
        policy_input = torch.cat([dist_to_trajectory, angle, torch.flatten(self.state[0, :])], dim=0).float().to(self.device)
        next_obs     = policy_input

        # renormalize state for next round of dynamics prediction
        self.state = self.normalize_state(self.state, self.device)
        info = {"delta" : delta, \
          "uncertain" : self.config.uncertainty_config.uncertainty_coeff * uncertain, \
          "predictions": all_predictions} 
        res = next_obs.cpu().detach().numpy(), float(reward_out[0].item()), bool(uncertain or timeout), {}
        return res

    # speed, steer
    def unnormalize_state(self, obs, device):
        return self.norm_stats["obs"]["std"].to(device) * obs + self.norm_stats["obs"]["mean"].to(device)

    def normalize_state(self, obs, device):
        return (obs - self.norm_stats["obs"]["mean"].to(device))/self.norm_stats["obs"]["std"].to(device)

    # Δx, Δy, Δyaw, Δspeed, Δsteer
    def unnormalize_delta(self, delta, device):
        return self.norm_stats["delta"]["std"].to(device) * delta + self.norm_stats["delta"]["mean"].to(device)

    def normalize_delta(self, delta, device):
        return (delta - self.norm_stats["delta"]["mean"].to(device))/self.norm_stats["delta"]["std"].to(device)
