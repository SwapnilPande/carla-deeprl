import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import scipy.spatial

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from data_modules import OfflineCarlaDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# speed, steer
def unnormalize_state(obs, device):
    return torch.tensor([0.1100, 0.1323]).to(device)*obs + torch.tensor([0.0196, -0.0887]).to(device)

def normalize_state(obs, device):
    return (obs - torch.tensor([0.0196, -0.0887]).to(device))/torch.tensor([0.1100, 0.1323]).to(device)

# Δx, Δy, Δyaw, Δspeed, Δsteer
def unnormalize_delta(delta, device):
    raise NotImplementedError

def normalize_delta(delta, device):
    raise NotImplementedError

''' 
Calculates L2 distance between waypoint, vehicle
@param waypoint:     torch.Tensor([x,y])
        vehicle_pose: torch.Tensor([x, y, yaw])
'''
def distance_vehicle(waypoint, vehicle_pose):
    vehicle_loc = vehicle_pose[:2] # get x, y 
    dist = torch.dist(waypoint, vehicle_loc).item()
    return dist


''' 
Calculates dot product, angle between vehicle, waypoint
@param waypoint:     [x, y]
       vehicle_pose: torch.Tensor([x, y, yaw])
'''
def get_dot_product_and_angle(vehicle_pose, waypoint):

    waypoint = torch.FloatTensor(waypoint)

    v_begin         = vehicle_pose[:2]
    vehicle_yaw     = vehicle_pose[2]  
    v_end = v_begin + torch.Tensor([torch.cos(torch.deg2rad(vehicle_yaw)), \
                                    torch.sin(torch.deg2rad(vehicle_yaw))])

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
        (torch.linalg.norm(w_vec) * torch.linalg.norm(v_vec)), -1.0, 1.0)))

    # make vectors 3D for cross product
    v_vec_3d = torch.hstack((v_vec, torch.Tensor([0])))
    w_vec_3d = torch.hstack((w_vec, torch.Tensor([0])))

    _cross = torch.cross(v_vec_3d, w_vec_3d)

    # if negative steer, turn left 
    if _cross[2] < 0:   
        angle *= -1.0
    
    # assert cross product a x b = |a||b|sin(angle)
    assert(torch.isclose(_cross[2], torch.norm(v_vec_3d) * torch.norm(w_vec_3d) * torch.sin(angle)))

    return dot, angle, w_vec

'''
Gets distance of vehicle to a line formed by two waypoints
@param waypoint1:     [x,y]
       waypoint2:     [x,y]
       vehicle_pose:  torch.Tensor([x,y, yaw])
'''
def vehicle_to_line_distance(vehicle_pose, waypoint1, waypoint2):
    waypoint1 = torch.FloatTensor(waypoint1)
    waypoint2 = torch.FloatTensor(waypoint2)

    vehicle_loc   = vehicle_pose[:2] # x, y coords
    
    a_vec = torch.sub(waypoint2, waypoint1)   # forms line between two waypoints
    b_vec = torch.sub(vehicle_loc, waypoint1) # line from vehicle to first waypoint
    
    # make 3d for cross product
    a_vec_3d = torch.hstack((a_vec, torch.Tensor([0])))
    b_vec_3d = torch.hstack((b_vec, torch.Tensor([0])))

    dist_vec = torch.cross(a_vec_3d, b_vec_3d) / torch.linalg.norm(a_vec_3d)
    return abs(dist_vec[2]) # dist

'''
Calculate dist to trajectory, angle
@param waypoints:    [torch.Tensor([wp1_x,wp1_y]), torch.Tensor([wp2_x,wp2_y)......]
       vehicle pose: torch.Tensor([x,y,yaw])
@returns dist_to_trajectory, angle, ...
'''
def process_waypoints(waypoints, vehicle_pose):

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
        dist_i = distance_vehicle(torch.FloatTensor(waypoint), vehicle_pose)
        if dist_i < min_dist:
            min_dist_index = i
            min_dist = dist_i

    wp_len = len(waypoints)
    if min_dist_index >= 0:
        # pop waypoints up untilthe one with min dist
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
        dot, angle, w_vec = get_dot_product_and_angle(vehicle_pose, waypoint)

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
        angle = np.mean(np.array(next_waypoints_angles))
    else:
        print("No next waypoint found!")
        angle = 0

    if len(next_waypoints) > 1:
        print(next_waypoints)
        # get dist from vehicle to a line formed by the next two wps 
        dist_to_trajectory = vehicle_to_line_distance(
                                vehicle_pose,
                                next_waypoints[0],
                                next_waypoints[1])

    # if only one next waypoint, use it and second to last 
    elif len(next_waypoints) > 0:
        dist_to_trajectory = vehicle_to_line_distance(
                                vehicle_pose,
                                second_last_waypoint,
                                next_waypoints[0])

    else:

        if second_last_waypoint and last_waypoint:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    second_last_waypoint,
                                    last_waypoint)
            
        else:
            dist_to_trajectory = 0

    return angle, dist_to_trajectory, next_waypoints, next_waypoints_angles, next_waypoints_vectors



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
        self.device = "cuda:{}".format(self.device_num) if torch.cuda.is_available() else "cpu"

        self.logger = logger
        # init
        self.state = None
        self.vehicle_pose = None
        self.waypoints = None


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

    # sample from dataset 
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


    ''' 
    Calculates reward according to distance to trajectory 
    @params next_state:   vehicle state after taking action
            dist_to_traj: distance from vehicle to current trajectory
    '''
    def calculate_reward(self, next_state, dist_to_traj):
        speed = next_state[0]*15
        dist_to_trajectory = dist_to_traj *10

        # reward for speed, penalize for straying from target trajectory 
        off_route = torch.abs(dist_to_trajectory) > 10
        return torch.unsqueeze(speed - 1.2*torch.abs(dist_to_trajectory) - (off_route * 50.), dim = 0)


    ''' Resets environment '''
    def reset(self, obs = None, action = None):
        print("Resetting environment...\n")
        if obs is None:
            # samples returns (obs, act, reward, delta, done, waypoints, vehicle_pose)
            # obs: [[speed_t, steer_t, delta_time_t], [speed_t-1, steer_t-1, delta_time_t-1], ...]
            obs, action, _, _, _, waypoints, vehicle_pose = self.sample()

            self.obs = torch.squeeze(obs)
            self.past_action = torch.squeeze(action)
            self.waypoints = torch.squeeze(waypoints)
            self.vehicle_pose = torch.squeeze(vehicle_pose)
            # state only includes speed, steer
            self.state =  self.obs[:, :2]

        print('obs', self.obs)
        print('action', self.past_action)
        print('waypoints', self.waypoints)
        print('vehicle pose', self.vehicle_pose)
        print('state', self.state)

        self.steps_elapsed = 0

        return self.state  # [speed_frames, steer_frames, delta_frames]

   
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

        print("Taking step...\n")

        # clamp new action to safe range
        new_action = torch.clamp(new_action, -1, 1).to(self.device)
        # insert new action at front, delete oldest action
        action = torch.cat([new_action, self.past_action[:-1]])

        # if obs not passed in
        if not obs:
            obs = self.obs
       
        ############ feed predictions (normalized) through dynamics model ##############

        # input [[speed_t, steer_t, Δtime_t, action_t], [speed_t-1, steer_t-1, Δt-1, action_t-1]]
    
        # TODO: dynamics model must flatten input to 1D list
        dynamics_input = torch.cat([obs, action.reshape(-1,1)], dim = 1).float()

        # dynamics_input = torch.flatten(torch.cat([obs, action.reshape(-1,1)], dim = 1).float())
        # predicts normalized deltas across models 
        predictions = torch.squeeze(self.dynamics.predict(dynamics_input))
        # randomly sample a model
        model_idx = np.random.choice(self.dynamics.n_models)
        # output delta: [Δx_t+1, Δy_t+1, Δtheta_t+1, Δspeed_t+1, Δsteer_t+1]
        delta = torch.clone(predictions[model_idx])
        delta = unnormalize_delta(delta, device) # TODO: FILL IN UNNORMALIZE

        # predicted change in x, y, th
        delta_vehicle_pose = delta[:3]
        # change in speed, steer 
        delta_state = delta[3:5]

        # update vehicle pose
        self.vehicle_pose = self.vehicle_pose + delta_vehicle_pose
        # update next state
        self.state = self.update_next_state(delta_state)
               

        ###################### calculate waypoint features (in global frame)  ##############################

        angle, dist_to_trajectory, _, _, _ = process_waypoints(self.waypoints, self.vehicle_pose)

        ################## calc reward with penalty for uncertainty ##############################

        reward_out = self.calculate_reward(self.state, dist_to_trajectory)

        uncertain = 0 # self.usad(predictions.cpu().numpy())    # TODO: FILL IN
        reward_out[0] = reward_out[0] - uncertain * 150
        reward_out = torch.squeeze(reward_out)

        # advance time 
        self.steps_elapsed += 1
        timeout = self.steps_elapsed >= self.timeout_steps

        # log 
        if(uncertain and self.logger is not None):
            # self.logger.get_metric("average_halt")
            self.logger.log_metrics({"halts" : 1})
        elif(timeout and self.logger is not None):
            self.logger.log_metrics({"halts" : 0})


        ######################### build policy input ##########################

        # Policy input (unnormalized): dist_to_trajectory, next orientation, speed, steer
        dist_to_trajectory = torch.Tensor([dist_to_trajectory])
        angle = torch.Tensor([angle])


        # take most recent state
        policy_input = torch.cat([dist_to_trajectory, angle, torch.flatten(self.state[0, :])], dim=0).float().to(self.device)
        next_obs = policy_input

        # renormalize state for next round of dynamics prediction
        self.state = normalize_state(self.state, device)

        return next_obs, reward_out, (uncertain or timeout), {"delta" : delta, "uncertain" : 100*uncertain}

