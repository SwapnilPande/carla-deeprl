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
def rot(theta):
    R = torch.Tensor([[ torch.cos(theta), -torch.sin(theta)],
                      [ torch.sin(theta), torch.cos(theta)]])
    return R

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



# def construct_obs(self,)


class OfflineCarlaDataset(Dataset):
    """ Offline dataset """

    def __init__(self,
                    path,
                    use_images=True,
                    frame_stack=1,
                    obs_dim = 2,
                    additional_state_dim = 0,
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

        # Keys to accesss values in the dataset
        steer_key = "control_steer"
        speed_key = "speed"
        wall_time_key = "wall_time"
        vehicle_x_key = "ego_vehicle_x"
        vehicle_y_key = "ego_vehicle_y"
        vehicle_theta_key = "ego_vehicle_theta"
        waypoints_key = "waypoints"
        reward_key = "reward"
        done_key = "done"
        action_key = "action"

        print("Loading data from: {}".format(path))
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
                        obs.append(torch.FloatTensor([samples[i-j][steer_key], samples[i-j][speed_key]]))

                        # Check that none of the observations are nan, break if nan is present
                        if np.isnan(obs[-1]).any():
                            print(trajectory_path)
                            break

                        # Additional state: delta time between each frame/timestep
                        # additional_state.append(torch.FloatTensor([samples[i-j+1]['wall'] - samples[i-j]['wall']]))

                        # Action taken
                        action.append(torch.FloatTensor(samples[i-j][action_key]))



                    # vehicle pose at timestep t
                    vehicle_pose_cur = torch.FloatTensor([samples[i+1][vehicle_x_key],
                                                          samples[i+1][vehicle_y_key],
                                                          samples[i+1][vehicle_theta_key]])

                    vehicle_pose_prev = torch.FloatTensor([samples[i][vehicle_x_key],
                                                           samples[i][vehicle_y_key],
                                                           samples[i][vehicle_theta_key]])

                    # split vehicle pose into location [x,y], and yaw
                    vehicle_loc_cur = vehicle_pose_cur[:2]
                    vehicle_loc_prev = vehicle_pose_prev[:2]
                    vehicle_theta_cur = vehicle_pose_cur[2]
                    vehicle_theta_prev = vehicle_pose_prev[2]


                    # homogeneous transform: get relative change in vehicle pose
                    global_loc_offset = vehicle_loc_cur - vehicle_loc_prev
                    vehicle_loc_delta = torch.inverse(rot(torch.deg2rad(vehicle_theta_prev))) @ global_loc_offset.unsqueeze(-1)

                    # Construct next state prediction delta
                    delta_x, delta_y = vehicle_loc_delta[:2]
                    delta_theta = vehicle_theta_cur - vehicle_theta_prev

                    if(delta_theta < -180):
                        delta_theta += 360
                    elif(delta_theta > 180):
                        delta_theta -= 360

                    assert delta_theta >= -180 and delta_theta <= 180

                    delta = torch.FloatTensor([delta_x,
                                                delta_y,
                                                delta_theta,
                                                samples[i+1][steer_key] - samples[i][steer_key],
                                                samples[i+1][speed_key] - samples[i][speed_key]])

                    # convert stacked frame list to torch tensor
                    self.obs.append(torch.stack(obs))
                    # self.additional_state.append(torch.stack(additional_state))
                    self.actions.append(torch.stack(action))
                    self.delta.append(delta)
                    self.vehicle_poses.append(vehicle_pose_cur)
                    # get waypoints for current timestep
                    waypoints = torch.FloatTensor(samples[i][waypoints_key])
                    self.waypoint_module.store_waypoints(waypoints)

                    # rewards, terminal at each timestep
                    self.rewards.append(torch.FloatTensor([samples[i][reward_key]]))
                    self.terminals.append(samples[i][done_key])

            self.obs = torch.stack(self.obs)
            self.actions = torch.stack(self.actions)
            # self.additional_state = torch.stack(self.additional_state)
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
        # additional_state = self.additional_state[idx]

        mlp_features = obs #torch.cat([obs, additional_state.reshape(-1,1)], dim = 1)
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


    # Updates dataset with newly-collected trajectories saved to new_path
    def update(self, new_path):
        self.setup(new_path)


    def setup(self, new_path = None):


        # If new_path passed in, simply add newly collected data to existing datasets
        if new_path is not None:
            # import pdb; pdb.set_trace();

            self.datasets.append(OfflineCarlaDataset(path=new_path, frame_stack=self.frame_stack))
            self.num_paths += 1
        else:
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

            traj_obs              = torch.vstack([d.obs[:,-1,:].squeeze() for d in self.datasets])
            traj_actions          = torch.vstack([d.actions[:,-1,:].squeeze() for d in self.datasets])
            # traj_additional_state = torch.vstack([d.additional_state[:,-1, :] for d in self.datasets])
            # no need to index because deltas are not stacked
            traj_delta            = torch.vstack([d.delta.squeeze() for d in self.datasets])


        
            # calculate mean, stdev across all trajectories
            self.normalization_stats["obs"]              = compute_mean_std(traj_obs,n)
            # self.normalization_stats["additional_state"] = compute_mean_std(traj_additional_state, n)
            self.normalization_stats["action"]           = compute_mean_std(traj_actions, n)
            self.normalization_stats["delta"]            = compute_mean_std(traj_delta, n)


        else:
            print('No normalization: Setting normalization stats to Mean=0, Std=1')
            self.normalization_stats["obs"] = {"mean" : torch.zeros((1, self.datasets[0].obs_dim)), "std" : torch.ones((1, self.datasets[0].obs_dim))}
            self.normalization_stats["action"] = {"mean" : torch.zeros((1, self.datasets[0].action_dim)), "std" : torch.ones((1, self.datasets[0].action_dim))}
            self.normalization_stats["delta"] = {"mean" : torch.zeros((1, self.datasets[0].state_dim_out)), "std" : torch.ones((1, self.datasets[0].state_dim_out))}  

            # print('obs',self.normalization_stats["obs"])
            # print('act', self.normalization_stats["action"])
            # print('delta', self.normalization_stats["delta"])

        # normalize
        for i in range(len(self.datasets)):
            self.datasets[i].obs= (self.datasets[i].obs - self.normalization_stats["obs"]["mean"]) / self.normalization_stats["obs"]["std"]
            # self.datasets[i].additional_state = (self.datasets[i].additional_state - self.normalization_stats["additional_state"]["mean"]) / self.normalization_stats["additional_state"]["std"]
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
    def train_dataloader(self, weighted = True, batch_size_override = None):
        weights = torch.ones(size = (len(self.train_data),))
        sampler = None
        if(weighted):
            for i in tqdm(range(len(self.train_data))):
                _, _, _, delta, _, _ = self.train_data[i]
                weight = torch.sqrt(torch.square(delta[...,3]) + torch.square(delta[...,4]))
                weights[i] = weights[i] + weight

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