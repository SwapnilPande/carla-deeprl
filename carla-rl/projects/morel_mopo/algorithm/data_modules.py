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


''' Instantiate WaypointModule for each dataset'''
class WaypointModule():
    ''' init waypoints '''
    def __init__(self):
        self.waypoints = []
    def store_waypoints(self, waypoints):
        self.waypoints.append(waypoints)
    def get_waypoints(self, idx):
        return self.waypoints[idx]


class OfflineCarlaDataset(Dataset):
    """ Offline dataset """

    def __init__(self,
                    path,
                    use_images=True,
                    frame_stack=1,
                    obs_dim = 2,
                    additional_state_dim = 1,
                    action_dim = 2,
                    state_dim_out = 5):

        if(frame_stack < 1):
            raise Exception("Frame stack must be greater than or equal to 1")

        trajectory_paths = glob.glob('{}/*'.format(path))
        # assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

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
        self.actions, self.rewards, self.terminals = [], [], []
        self.vehicle_poses = []
        self.red_light = []

        self.waypoint_module = WaypointModule()


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
               
                    # convert stacked frame list to torch tensor
                    self.obs.append(torch.stack(obs))
                    self.additional_state.append(torch.stack(additional_state))
                    self.actions.append(torch.stack(action))
                    self.delta.append(delta)
                    self.vehicle_poses.append(vehicle_pose_cur)
                    # get waypoints for current timestep
                    waypoints = torch.FloatTensor(samples[i]['waypoints'])
                    self.waypoint_module.store_waypoints(waypoints)
            
                    # rewards, terminal at each timestep
                    self.rewards.append(torch.FloatTensor([samples[i]['reward']]))
                    self.terminals.append(samples[i]['done'])

            self.obs = torch.stack(self.obs)
            self.actions = torch.stack(self.actions)
            self.additional_state = torch.stack(self.additional_state)
            self.delta = torch.stack(self.delta)
            self.vehicle_poses = torch.stack(self.vehicle_poses)


    def __getitem__(self, idx):
        '''
        obs:              B x F x 2
        additional_state: B x F x 1
        mlp_features:     B x F x 3
        action:           B x F x 2
        reward:           B x 1
        delta:            B x 5
        done:             B x 1
        vehicle_pose:     B x 3
        '''

        obs = self.obs[idx]
        additional_state = self.additional_state[idx]

        mlp_features = torch.cat([obs, additional_state.reshape(-1,1)], dim = 1)
        action = self.actions[idx]
        reward = self.rewards[idx]
        delta = self.delta[idx]
        done = self.terminals[idx]
        vehicle_pose = self.vehicle_poses[idx]

        return mlp_features, action, reward, delta, done, vehicle_pose

    def __len__(self):
        return len(self.rewards)

    ''' randomly sample from dataset '''
    def sample_with_waypoints(self, idx):
        return self[idx], self.waypoint_module.get_waypoints(idx)


class OfflineCarlaDataModule():
    """ Datamodule for offline driving data """
    def __init__(self, cfg, normalize_data = True):
        super().__init__()
        self.normalize_data = normalize_data
        self.paths = cfg.dataset_paths
        self.num_paths = len(self.paths)
        self.batch_size = cfg.batch_size
        self.frame_stack = cfg.frame_stack
        self.num_workers = cfg.num_workers
        self.train_val_split = cfg.train_val_split
        self.datasets = None
        self.train_data = None
        self.val_data = None
        self.normalization_stats = {
            "obs": None,
            "additional_state" : None,
            "delta" : None,
            "action" : None
        }

    def setup(self):
        # Create a dataset for each trajectory
        self.datasets = [OfflineCarlaDataset(path=path, frame_stack=self.frame_stack) for path in self.paths]
        # Dimensions
        self.state_dim_in = self.datasets[0].obs_dim + self.datasets[0].additional_state_dim
        self.state_dim_out = self.datasets[0].state_dim_out
        self.action_dim = self.datasets[0].action_dim
        self.frame_stack = self.frame_stack

        # normalize across all trajectories
        if self.normalize_data:
            # number of total timesteps 
            n = sum(len(d.rewards) for d in self.datasets)
            
            traj_obs              = torch.vstack([d.obs[:,-1,:] for d in self.datasets])
            traj_actions          = torch.vstack([d.actions[:,-1,:] for d in self.datasets])
            traj_additional_state = torch.vstack([d.additional_state[:,-1, :] for d in self.datasets])
            # no need to index because deltas are not stacked
            traj_delta            = torch.vstack([d.delta for d in self.datasets])


            # calculate mean, stdev across all trajectories
            self.normalization_stats["obs"]              = compute_mean_std(traj_obs,n)
            self.normalization_stats["additional_state"] = compute_mean_std(traj_additional_state, n)
            self.normalization_stats["action"]           = compute_mean_std(traj_actions, n)
            self.normalization_stats["delta"]            = compute_mean_std(traj_delta, n)

            # normalize
            for i in range(len(self.datasets)):
                self.datasets[i].obs= (self.datasets[i].obs - self.normalization_stats["obs"]["mean"]) / self.normalization_stats["obs"]["std"]
                self.datasets[i].additional_state = (self.datasets[i].additional_state - self.normalization_stats["additional_state"]["mean"]) / self.normalization_stats["additional_state"]["std"]
                self.datasets[i].actions =  (self.datasets[i].actions - self.normalization_stats["action"]["mean"]) / self.normalization_stats["action"]["std"]
                self.datasets[i].delta = (self.datasets[i].delta - self.normalization_stats["delta"]["mean"]) / self.normalization_stats["delta"]["std"]




        # concat datasets across all trajectories (used for dynamics training)
        self.concat_dataset = torch.utils.data.ConcatDataset(self.datasets)

        # split into train, val datasets for dynamics training
        train_size = int(len(self.concat_dataset) * self.train_val_split)
        val_size = len(self.concat_dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.concat_dataset, (train_size, val_size))
 
    ''' This is used in FakeEnv for dynamics evaluation (with waypoints, batch size = 1) '''
    def sample_with_waypoints(self):
        # selects a trajectory path
        path_idx = np.random.randint(self.num_paths)
        dataset = self.datasets[path_idx]
        
        # selects a timestep along trajectory to sample from 
        num_timesteps = len(dataset)
        idx = np.random.randint(num_timesteps)
        return dataset.sample_with_waypoints(idx)


    ''' This is used for dynamics training (no waypoint input needed, batch size set)'''
    def train_dataloader(self, weighted = False, batch_size_override = None):
        weights = torch.ones(size = (len(self.train_data),))
        sampler = None
        if(weighted):
            for i in range(len(self.train_data)):
                _, _, _, deltas, _, _,  = self.train_data[i]
                weights[i] = torch.abs(deltas[0]) + 0.1

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
        else:

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
                            batch_size=batch_size,
                            num_workers=self.num_workers,
                            sampler = sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data,\
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)


"""
Online dataset handling
"""
