""" PyTorch datasets for offline RL experiments """

import glob
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from tqdm import tqdm

"""
Offline dataset handling
"""

''' Rotation about z axis
@param theta
@return: rotation matrix about z axis by theta
'''
def rotz(theta):
    Rz = torch.Tensor([[ torch.cos(theta), -torch.sin(theta), 0 ],
                      [ torch.sin(theta), torch.cos(theta) , 0 ],
                      [ 0,              0,             1]])
    return Rz

''' Computes the mean and standard deviation of data
@param data across all trajectories
       n: number of data points
@returns: dictionary storing mean, std
'''
def compute_mean_std(data, n):
    # sum data row-wise across all trajectories
    data_sum    = torch.sum(data, dim=0)
    data_sum_sq = torch.sum(torch.square(data), 0)
    mean        = data_sum/n
    std         =  np.sqrt(data_sum_sq/n - torch.square(mean))
    return {"mean" : mean, "std" : std} 


class OfflineCarlaDataset(Dataset):
    """ Offline dataset """

    def __init__(self,
                    path,
                    use_images=True,
                    frame_stack=1,
                    normalize_data = True,
                    obs_dim = 2,
                    additional_state_dim = 1,
                    action_dim = 2,
                    state_dim_out = 5):

        if(frame_stack < 1):
            raise Exception("Frame stack must be greater than or equal to 1")

        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.use_images = use_images

        # Save data shape
        self.frame_stack = frame_stack
        self.obs_dim = obs_dim
        self.additional_state_dim = additional_state_dim
        self.action_dim = action_dim
        self.state_dim_out = state_dim_out

        # obs: features predicted for the next time step
        # additional_state: change in wall time
        # delta: change in x, y, yaw, speed, steer
        self.obs, self.additional_state, self.delta = [], [], []
        self.waypoints = []

        self.actions, self.rewards, self.terminals = [], [], []
        self.vehicle_poses = []
        self.red_light = []

        self.normalization_stats = {
            "obs": None,
            "additional_state" : None,
            "delta" : None,
            "action" : None
        }

    
        print("Loading data")

        # Don't calculate gradients for descriptive statistics
        with torch.no_grad():
            # Loop over all trajectories
            for trajectory_path in tqdm(trajectory_paths):
                samples = []
                json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
                traj_length = len(json_paths)

                # Loop over files and load in each timestep (represented by a single json file) for each trajectory
                for i in range(traj_length):
                    with open(json_paths[i]) as f:
                        sample = json.load(f)

                    samples.append(sample)

                # Exit if the trajectory is too short
                if traj_length <= (self.frame_stack + 1):
                    continue

                # Construct observations across all timesteps i in trajectory
                for i in range(self.frame_stack - 1, traj_length-1):
                    # Frame stacks for each element
                    obs = []
                    additional_state = []
                    action = []
                    delta = []

                    # at each timestep, collect observations for last <frame_stack> frames
                    for j in range(self.frame_stack):

                        # [[speed_t, steer_t], [speed_t-1, steer_t-1], ...]
                        obs.append(torch.FloatTensor([samples[i-j]['speed'], samples[i-j]['steer']]))

                        # Check that none of the observations are nan, break if nan is present
                        if np.isnan(obs[-1]).any():
                            print(trajectory_path)
                            break

                        # Additional state: delta time between each frame/timestep
                        additional_state.append(torch.FloatTensor([samples[i-j+1]['wall'] - samples[i-j]['wall']]))

                        # Action taken
                        action.append(torch.FloatTensor(samples[i-j]['action']))



                    # vehicle pose at timestep t
                    vehicle_pose_cur = torch.FloatTensor([samples[i+1]['x'],
                                                          samples[i+1]['y'],
                                                          samples[i+1]['theta']])

                    vehicle_pose_prev = torch.FloatTensor([samples[i]['x'],
                                                           samples[i]['y'],
                                                           samples[i]['theta']])

                    # split vehicle pose into location [x,y], and yaw
                    vehicle_loc_cur = vehicle_pose_cur[:2]
                    vehicle_loc_prev = vehicle_pose_prev[:2]
                    vehicle_theta_cur = vehicle_pose_cur[2]
                    vehicle_theta_prev = vehicle_pose_prev[2]


                    # homogeneous transform: get relative change in vehicle pose
                    global_loc_offset = torch.cat([vehicle_loc_cur, torch.Tensor([1])], dim=0) - torch.cat([vehicle_loc_prev, torch.Tensor([1])], dim=0)
                    vehicle_loc_delta = torch.inverse(rotz(torch.deg2rad(vehicle_theta_prev))) @ global_loc_offset.unsqueeze(-1)

                    # Construct next state prediction delta
                    delta_x, delta_y = vehicle_loc_delta[:2]
                    delta_theta = vehicle_theta_cur - vehicle_theta_prev
                    delta = torch.FloatTensor([delta_x,
                                                delta_y,
                                                delta_theta,
                                                samples[i+1]['speed'] - samples[i]['speed'],
                                                samples[i+1]['steer'] - samples[i]['steer']])
                    # get waypoints
                    waypoints = torch.FloatTensor(samples[i]['waypoints'])


                    # convert stacked frame list to torch tensor
                    self.obs.append(torch.stack(obs))
                    self.additional_state.append(torch.stack(additional_state))
                    self.actions.append(torch.stack(action))
                    self.delta.append(delta)
                    self.waypoints.append(waypoints)
                    self.vehicle_poses.append(vehicle_pose_cur)

                    # rewards, terminal at each timestep
                    self.rewards.append(torch.FloatTensor([samples[i]['reward']]))
                    self.terminals.append(samples[i]['done'])
            
            self.obs = torch.stack(self.obs)
            self.actions = torch.stack(self.actions)
            self.additional_state = torch.stack(self.additional_state)
            self.delta = torch.stack(self.delta)
            self.vehicle_poses = torch.stack(self.vehicle_poses)

            # normalize using z-score 
            if normalize_data:
                n = len(self.rewards)

                # get last frame from each timestep
                traj_obs              = self.obs[:,-1,:] 
                traj_actions          = self.actions[:,-1,:]
                traj_additional_state = self.additional_state[:,-1, :] 
                # no need to index because deltas are not stacked
                traj_delta            = self.delta

                # calculate mean, stdev across all trajectories 
                self.normalization_stats["obs"]              = compute_mean_std(traj_obs,n)
                self.normalization_stats["additional_state"] = compute_mean_std(traj_additional_state, n)
                self.normalization_stats["action"]           = compute_mean_std(traj_actions, n)
                self.normalization_stats["delta"]            = compute_mean_std(traj_delta, n)

                # normalize
                self.obs = (self.obs - self.normalization_stats["obs"]["mean"]) / self.normalization_stats["obs"]["std"]
                self.additional_state = (self.additional_state - self.normalization_stats["additional_state"]["mean"]) / self.normalization_stats["additional_state"]["std"]
                self.actions =  (self.actions - self.normalization_stats["action"]["mean"]) / self.normalization_stats["action"]["std"]
                self.delta = (self.delta - self.normalization_stats["delta"]["mean"]) / self.normalization_stats["delta"]["std"]


    def __getitem__(self, idx):
        '''
        obs:              B x F x 2
        additional_state: B x F x 1
        mlp_features:     B x F x 3 
        action:           B x F x 2   
        reward:           B x 1           
        done:             B x 1           
        vehicle_pose:     B x 3         
        delta:            B x 5           
        '''

        obs = self.obs[idx]
        additional_state = self.additional_state[idx]

        mlp_features = torch.cat([obs, additional_state.reshape(-1,1)], dim = 1)
        waypoints = self.waypoints[idx]
        action = self.actions[idx]
        delta = self.delta[idx]
        reward = self.rewards[idx]
        done = self.terminals[idx]
        vehicle_pose = self.vehicle_poses[idx]

        return mlp_features, action, reward, delta, done, waypoints, len(waypoints), vehicle_pose

    def __len__(self):
        return len(self.rewards)

    ''' randomly sample from dataset '''
    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]


''' Pads waypoints with zeros so that all waypoints sequences are same length
@param: batch of data
@out:   original features but with padded waypoints
'''
def collate_fn(batch):
    mlp_features, action, reward, delta, done, waypoints, num_wps_list, vehicle_pose = [list(x) for x in zip(*batch)]
    # waypoint feature is at the fifth index
    wp_idx = 5
    # get maximum number of waypoints across all sequences
    max_wps = max(num_wps_list)
    # get innermost dimension of each wp
    wp_dim = batch[0][wp_idx].size(-1)
    # create padding
    padded_waypoints = torch.zeros((len(batch), max_wps, wp_dim))
    for i in range(len(waypoints)):
        wp = waypoints[i]
        num_wps, wp_dim = batch[i][wp_idx].size()
        padded_waypoints[i] = torch.cat([wp, torch.zeros((max_wps - num_wps, wp_dim))])

    # convert to tensors
    mlp_features = torch.stack(mlp_features)
    action       = torch.stack(action)
    delta        = torch.stack(delta)
    vehicle_pose = torch.stack(vehicle_pose)

    new_batch = mlp_features, action, reward, delta, done, padded_waypoints, num_wps_list, vehicle_pose
    return new_batch



class OfflineCarlaDataModule():
    """ Datamodule for offline driving data """

    def __init__(self, cfg):
        super().__init__()
        self.paths = cfg.dataset_paths
        self.batch_size = cfg.batch_size
        self.frame_stack = cfg.frame_stack
        self.num_workers = cfg.num_workers
        self.train_val_split = cfg.train_val_split

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self):
        # Create a dataset for each trajectory
        datasets = [OfflineCarlaDataset(path=path, frame_stack=self.frame_stack) for path in self.paths]
        # Create a single dataset that concatenates all of the trajectory datasets
        self.dataset = torch.utils.data.ConcatDataset(datasets)

        #TODO Decide the proper order in which to cinfugre this
        self.state_dim_in = datasets[0].obs_dim + datasets[0].additional_state_dim
        self.state_dim_out = datasets[0].state_dim_out
        self.action_dim = datasets[0].action_dim
        self.frame_stack = self.frame_stack

        # self.dataset = OfflineCarlaDataset(use_images=self.use_images, path=self.path, frame_stack=self.frame_stack)
        train_size = int(len(self.dataset) * self.train_val_split)
        val_size = len(self.dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))

    def train_dataloader(self, weighted = False, batch_size_override = None):
        weights = torch.ones(size = (len(self.train_data),))

        sampler = None
        if(weighted):
            for i in range(len(self.train_data)):
                _, _, _, deltas, _, _, _, _ = self.train_data[i]
                weights[i] = torch.abs(deltas[0]) + 0.1

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
        else:
            #for i in range(len(self.train_data)):
             #   mlp_features, _, _, _, red_light = self.train_data[i]
             #   weights[i] = torch.pow(torch.abs(mlp_features[4]), 0.5) + 0.1

            #    if(red_light[0] == 1):
            #        weights[i] = 0

            sampler = None #torch.utils.data.WeightedRandomSampler(
            #    weights=weights,
            #    num_samples=len(weights),
            #    replacement=True
            #)

        if(batch_size_override is None):
            batch_size = self.batch_size
        else:
            batch_size = batch_size_override

        return DataLoader(self.train_data,
                            collate_fn=collate_fn,
                            batch_size=batch_size,
                            num_workers=self.num_workers,
                            sampler = sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data,\
                          collate_fn=collate_fn, \
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


"""
Online dataset handling
"""


