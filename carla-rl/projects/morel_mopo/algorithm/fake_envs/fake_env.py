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



class NormalizedTensor:
    def __init__(self, mean, std, device):
        self.device = device

        if(not isinstance(mean, torch.Tensor)):
            if(not isinstance(mean, (np.ndarray, list))):
                raise Exception("Mean must be torch tensor, list, or numpy array")

            mean = torch.FloatTensor(mean)

        if(not isinstance(std, torch.Tensor)):
            if(not isinstance(std, (np.ndarray, list))):
                raise Exception("Std must be torch tensor, list, or numpy array")

            std = torch.FloatTensor(std)

        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

        self.dim = self.mean.shape[-1]
        if(self.dim != self.std.shape[-1]):
            raise Exception("Mean and Std dimensions are different")

        self._unnormalized_array = None

    @property
    def unnormalized(self):
        return self._unnormalized_array

    @unnormalized.setter
    def unnormalized(self, unnormalized_arr):
        if(not isinstance(unnormalized_arr, torch.Tensor)):
            if(not isinstance(unnormalized_arr, (np.ndarray, list))):
                raise Exception("Input must be torch tensor, list, or numpy array")

            unnormalized_arr = torch.FloatTensor(unnormalized_arr)

        if(self.dim != unnormalized_arr.shape[-1]):
            raise Exception("Dimension of input tensor ({}) does not match dimension of mean ({})".format(unnormalized_arr.shape[-1],
                                                                                                            self.dim))

        unnormalized_arr = unnormalized_arr.to(self.device)

        self._unnormalized_array = unnormalized_arr

    @property
    def normalized(self):
        if(self._unnormalized_array is None):
            return None

        return self.normalize_array(self._unnormalized_array)

    @normalized.setter
    def normalized(self, normalized_arr):
        if(not isinstance(normalized_arr, torch.Tensor)):
            if(not isinstance(normalized_arr, (np.ndarray, list))):
                raise Exception("Input must be torch tensor, list, or numpy array")

            normalized_arr = torch.FloatTensor(normalized_arr)

        if(self.dim != normalized_arr.shape[-1]):
            raise Exception("Dimension of input tensor ({}) does not match dimension of mean ({})".format(
                normalized_arr.shape[-1],
                self.dim))

        normalized_arr = normalized_arr.to(self.device)

        self._unnormalized_array = self.unnormalize_array(normalized_arr)

    def normalize_array(self, array):
        return (array - self.mean)/self.std

    def unnormalize_array(self, array):
        return self.std * array + self.mean

def rot(theta):
    R = torch.Tensor([[ torch.cos(theta), -torch.sin(theta)],
                      [ torch.sin(theta), torch.cos(theta)]])
    return R


'''
Calculates L2 distance between waypoint, vehicle
@param waypoint:     torch.Tensor([x,y])
        vehicle_pose: torch.Tensor([x, y, yaw])
'''
def distance_vehicle(waypoint, vehicle_pose, device):
    if not torch.is_tensor(waypoint):
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

    if torch.allclose(waypoint1, waypoint2):
        distance = distance_vehicle(waypoint1, vehicle_pose, device)
        return abs(distance)


    vehicle_loc   = vehicle_pose[:2] # x, y coords

    a_vec = torch.sub(waypoint2, waypoint1)   # forms line between two waypoints
    b_vec = torch.sub(vehicle_loc, waypoint1) # line from vehicle to first waypoint

    # make 3d for cross product
    a_vec_3d = torch.hstack((a_vec, torch.Tensor([0]).to(device)))
    b_vec_3d = torch.hstack((b_vec, torch.Tensor([0]).to(device)))

    dist_vec = torch.cross(a_vec_3d, b_vec_3d) / torch.linalg.norm(a_vec_3d)
    distance =  abs(dist_vec[2]) # dist
    # print("Distance: ", distance, '\n')
    return distance

'''
Calculate dist to trajectory, angle
@param waypoints:    [torch.Tensor([wp1_x,wp1_y]), torch.Tensor([wp2_x,wp2_y)......]
       vehicle pose: torch.Tensor([x,y,yaw])
@returns dist_to_trajectory, angle, ...
'''
def process_waypoints(waypoints, vehicle_pose, device):
    vehicle_pose.to(device)

    waypoints = waypoints[:, :2]
    waypoints = waypoints.tolist()

    next_waypoints_angles = []
    next_waypoints_vectors = []
    next_waypoints = []
    num_next_waypoints = 5
    last_waypoint, second_last_waypoint = None, None


    # closest wp to car
    min_dist_index = -1

    # Minimum distance to waypoint before we delete it
    # This number is taken from GlobalPlanner
    MIN_REMOVE_DISTANCE = 1.8

    for i, waypoint in enumerate(waypoints):
        # find wp that yields min dist between itself and car
        dist_i = distance_vehicle(waypoint, vehicle_pose, device)
        # print(f'wp {i},  {waypoint}, dist: {dist_i}')
        if dist_i <= MIN_REMOVE_DISTANCE:
            min_dist_index = i

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

    remaining_waypoints = waypoints
    # only keep next N waypoints
    for i, waypoint in enumerate(waypoints[:num_next_waypoints]):
        # dist to waypoint
        dot, angle, w_vec = get_dot_product_and_angle(vehicle_pose, waypoint, device)

        if len(next_waypoints_angles) == 0:
            next_waypoints_angles = [angle]
            next_waypoints = [waypoint]
            next_waypoints_vectors = [w_vec]
        else:
            # add back waypoints
            next_waypoints_angles.append(angle)
            next_waypoints.append(waypoint)
            next_waypoints_vectors.append(w_vec)

    # get mean of all angles to figure out which direction to turn
    if len(next_waypoints_angles) > 0:
        angle = torch.mean(torch.FloatTensor(next_waypoints_angles))
    else:
        # print("No next waypoint found!")
        angle = 0



    if len(next_waypoints) > 1:
        # get dist from vehicle to a line formed by the next two wps
        dist_to_trajectory = vehicle_to_line_distance(
                                vehicle_pose,
                                next_waypoints[0],
                                next_waypoints[1],
                                device)

    # if only one next waypoint, use it and second to last
    elif len(next_waypoints) == 1:
        if second_last_waypoint:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    second_last_waypoint,
                                    next_waypoints[0],
                                    device)
        else:
            print("CODE BROKE HERE UH OH _----------------------")
            dist_to_trajectory = 0.0

    else: # Run out of wps
        if second_last_waypoint and last_waypoint:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    second_last_waypoint,
                                    last_waypoint,
                                    device)

        else:
            dist_to_trajectory = 0.0

    return angle, dist_to_trajectory, next_waypoints, next_waypoints_angles, next_waypoints_vectors, remaining_waypoints



class FakeEnv(gym.Env):
    def __init__(self, dynamics,
                        config,
                        logger = None):

        # Save logger
        self.logger = logger

        # Save and verify config
        self.config = config
        print('fake env config', self.config)
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
        self.state = NormalizedTensor(self.norm_stats["obs"]["mean"], self.norm_stats["obs"]["std"], self.device)
        self.past_action = NormalizedTensor(self.norm_stats["action"]["mean"], self.norm_stats["action"]["std"], self.device)
        self.deltas = NormalizedTensor(self.norm_stats["delta"]["mean"], self.norm_stats["delta"]["std"], self.device)

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
        self.past_action.unnormalized = torch.squeeze(action)
        self.waypoints = torch.squeeze(waypoints).to(self.device)
        self.vehicle_pose = torch.squeeze(vehicle_pose).to(self.device)

        self.steps_elapsed = 0

        self.model_idx = np.random.choice(self.dynamics.n_models)
        # Reset hidden state

        #TODO Return policy features, not dynamics features
        policy_obs, _, _ = self.get_policy_obs()
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
    def update_next_state(self, delta_state):
        # import ipdb; ipdb.set_trace()
        # calculate newest state
        newest_state = self.state.unnormalized[0, :] + delta_state

        # insert newest state at front
        self.state.unnormalized = torch.cat([newest_state.unsqueeze(0), self.state.unnormalized[:-1, :]], dim=0)

    def get_policy_obs(self):
        angle, dist_to_trajectory, next_waypoints, _, _, remaining_waypoints = process_waypoints(self.waypoints, self.vehicle_pose, self.device)

        # convert to tensors
        self.waypoints = torch.FloatTensor(remaining_waypoints)
        dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(self.device)
        angle              = torch.Tensor([angle]).to(self.device)

        return torch.cat([dist_to_trajectory, angle, torch.flatten(self.state.unnormalized[0, :])], dim=0).float().to(self.device), dist_to_trajectory, angle

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

            # Retrieve past actions
            action = self.past_action.unnormalized

            # clamp new action to safe range
            new_action = torch.squeeze(torch.clamp(new_action, -1, 1)).to(self.device)
            # import ipdb; ipdb.set_trace()

            # insert new action at front, delete oldest action
            self.past_action.unnormalized = torch.cat([new_action.unsqueeze(0), action[:-1, :]])

            # print("State")
            # print(self.state.unnormalized)

            # print("past action")
            # print(self.past_action.unnormalized)


            ############ feed obs, action into dynamics model for prediction ##############

            # input [[speed_t, steer_t, Δtime_t, action_t], [speed_t-1, steer_t-1, Δt-1, action_t-1]]
            # unsqueeze to form batch dimension for dynamics input

            # Get predictions across all models
            all_predictions = torch.stack(self.dynamics.predict(self.state.normalized, self.past_action.normalized)).squeeze(dim = 1)

            # Delta: prediction from one randomly selected model
            # [Δx_t+1, Δy_t+1, Δtheta_t+1, Δspeed_t+1, Δsteer_t+1]
            # import ipdb; ipdb.set_trace()
            self.deltas.normalized = torch.clone(all_predictions)

            # predicted change in x, y, th
            delta_vehicle_poses = self.deltas.unnormalized[:,:3]

            # change in steer, speed
            delta_state        = self.deltas.unnormalized[self.model_idx,3:5]

            # update vehicle pose
            vehicle_loc_delta = torch.transpose(torch.tensordot(rot(torch.deg2rad(self.vehicle_pose[2])).to(self.device), delta_vehicle_poses[:,0:2], dims = ([1], [1])), 0, 1)

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
            self.update_next_state(delta_state)

            ###################### calculate waypoint features (in global frame)  ##############################

            # self.waypoints = self.waypoints[:, :2] # get x,y coords for each waypoint
            # angle, dist_to_trajectory, next_waypoints, _, _, remaining_waypoints = process_waypoints(self.waypoints, self.vehicle_pose, self.device)

            policy_obs, dist_to_trajectory, angle = self.get_policy_obs()

            # check if at goal
            at_goal = (dist_to_trajectory == 0.0)

            ################## calc reward with penalty for uncertainty ##############################

            reward_out = compute_reward(self.state.unnormalized[0], dist_to_trajectory, self.config)

            uncertain =  self.usad(all_predictions.detach().cpu().numpy())
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

            res = policy_obs.cpu().numpy(), float(reward_out.item()), timeout or at_goal, info
            return res