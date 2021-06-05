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

''' Computes the mean and standard deviation
@param data_sum: sum of the inputs
       data_sum_sq: sum squared of the inputs
       n: number of data points
@returns: dictionary storing mean, std
'''
def compute_mean_std(data_sum, data_sum_sq, n):
    mean = data_sum/n
    std = torch.sqrt(data_sum_sq/n - mean**2)

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

        # Used to store running count of statistic for each variable
        if(normalize_data):
            obs_sum = torch.zeros((obs_dim,)).float()
            obs_sum_sq = torch.zeros((obs_dim,)).float()

            additional_state_sum = torch.zeros((additional_state_dim,)).float()
            additional_state_sum_sq = torch.zeros((additional_state_dim,)).float()

            delta_sum = torch.zeros((state_dim_out,)).float()
            delta_sum_sq = torch.zeros((state_dim_out,)).float()

            action_sum = torch.zeros((action_dim,)).float()
            action_sum_sq = torch.zeros((action_dim,)).float()

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


                    # TODO: The last few states in each trajectory is not factored into computing the descriptive statistics (specifically skipping (frame_stack -1 + 1) frames)
                    #       This is not a huge issue, because the length of trajectories is much greater than the number of trajectories
                    #       This might be worth fixing in the long term\
                    if(normalize_data):
                        obs_sum += obs[-1]
                        obs_sum_sq += obs[-1]**2

                        additional_state_sum += additional_state[-1]
                        additional_state_sum_sq += additional_state[-1]**2

                        action_sum += action[-1]
                        action_sum_sq += action[-1]**2

                        delta_sum += delta
                        delta_sum_sq += delta**2

                    # Convert stacked frame list to torch tensor
                    self.obs.append(torch.stack(obs))
                    self.additional_state.append(torch.stack(additional_state))
                    self.actions.append(torch.stack(action))
                    self.delta.append(delta)
                    self.waypoints.append(torch.FloatTensor(samples[i]['waypoints']))
                    self.vehicle_poses.append(vehicle_pose_cur)

                    # rewards, terminal at each timestep
                    self.rewards.append(samples[i]['reward'])
                    self.terminals.append(samples[i]['done'])

            # Compute descriptive statistics
            # Normalization across all trajectories
            if(normalize_data):
                self.normalization_stats["obs"] = compute_mean_std(obs_sum, obs_sum_sq, len(self.rewards))
                self.normalization_stats["additional_state"] = compute_mean_std(additional_state_sum, additional_state_sum_sq, len(self.rewards))
                self.normalization_stats["action"] = compute_mean_std(action_sum, action_sum_sq, len(self.rewards))
                self.normalization_stats["delta"] = compute_mean_std(delta_sum, delta_sum_sq, len(self.rewards))
            else:
                #TODO Fill with standard normal distribution N(0,1)
                pass


            # Normalize all data
            if(normalize_data):
                print("Normalizing all data")
                for i in tqdm(range(len(self.obs))):
                    # Tile mean and std frame_stack times to get the correct dim
                    self.obs[i] = (self.obs[i] - torch.unsqueeze(self.normalization_stats["obs"]["mean"], dim = 0))/torch.unsqueeze(self.normalization_stats["obs"]["std"], dim = 0)
                    self.additional_state[i] = (self.additional_state[i] - torch.unsqueeze(self.normalization_stats["additional_state"]["mean"], dim = 0))/torch.unsqueeze(self.normalization_stats["additional_state"]["std"], dim = 0)
                    self.actions[i] = (self.actions[i] - torch.unsqueeze(self.normalization_stats["action"]["mean"], dim = 0))/torch.unsqueeze(self.normalization_stats["action"]["std"], dim = 0)

                    # Don't need to tile delta stats because delta is never stacked
                    self.delta[i] = (self.delta[i] - self.normalization_stats["delta"]["mean"])/self.normalization_stats["delta"]["std"]

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        '''
        mlp_features: (<frame_stack>,3) [[speed_t, steer_t, Δtime_t], [speed_t-1, steer_t-1, Δtime_t-1], ...... ]
        waypoints:    (<traj_len>,2)    [[wp1x, wp1y], [wp2_x, wp2_y], [wp3_x, wp_3y].....[wpT_x, wpT_y]]
        action:       (<frame_stack>)   [a_t, a_t-1, a_t-2, ....]
        reward:                         reward_t
        done:                           done_t
        vehicle_pose:                   [x, y, θ]
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

        # print(mlp_features.shape)
        # print(action.shape)
        # # print(reward.shape)
        # print(delta.shape)
        # # print(done.shape)



        return mlp_features, action, reward, delta, done#, waypoints, vehicle_pose

    def __len__(self):
        return len(self.rewards)

    ''' randomly sample from dataset '''
    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]


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
                _, _, _, deltas, _ = self.train_data[i]
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
                            batch_size=batch_size,
                            num_workers=self.num_workers,
                            sampler = sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


"""
Online dataset handling
"""


