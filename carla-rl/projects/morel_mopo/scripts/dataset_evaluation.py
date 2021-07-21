import sys
import os
import argparse
import numpy as np

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from projects.morel_mopo.config.logger_config import CometLoggerConfig
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, DefaultGRUDynamicsConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
from projects.morel_mopo.algorithm.dynamics_gru import DynamicsGRUEnsemble
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
# from projects.morel_mopo.algorithm.fake_env import FakeEnv
# from projects.morel_mopo.config.fake_env_config import DefaultMainConfig

import matplotlib.pyplot as plt



import torch

def setup_log_dir(log_dir):
    print("Writing figures to directory {}".format(log_dir))
    if(not os.path.isdir(log_dir)):
        print("Log directory does not exist, creating now")
        os.makedirs(log_dir)

    # Data already exists in folder, decide if we want to continue
    if(os.listdir(log_dir) != []):
        print("WARNING: Log directory is not empty, this may result in data being overwritten")

def scatter(x, y, xlabel, ylabel, title, log_dir, f_name):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    file = os.path.join(log_dir, f_name)
    plt.savefig(file)
    plt.close()

def hist(x, bins, xlabel, ylabel, title, log_dir, f_name):
    plt.figure()
    plt.hist(x, bins = bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    file = os.path.join(log_dir, f_name)
    plt.savefig(file)
    plt.close()

def main(args):
    setup_log_dir(args.log_dir)

    class TempDataModuleConfig():
        def __init__(self):
            # Mixed Dataset
            self.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_noisy_policy",
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy"
                                ]
            self.batch_size = 1
            self.frame_stack = 1
            self.num_workers = 2
            self.train_val_split = 1.0
            self.normalize_data = False
            # Whether or not to stack the output
            # Use if the dynamics model is an RNN
            # False if no stack (MLP), True if stack (RNN)

    # data config
    data_config = TempDataModuleConfig()
    data_module = OfflineCarlaDataModule(data_config)
    data_module.setup()

    # Retrieve all the data from the dataset
    all_obs              = np.vstack([d.obs[:,-1,:].squeeze().cpu().numpy() for d in data_module.datasets])
    all_actions          = np.vstack([d.actions[:,-1,:].squeeze().cpu().numpy() for d in data_module.datasets])
    all_delta            = np.vstack([d.delta.squeeze().cpu().numpy() for d in data_module.datasets])

    print("DATASET SIZE: {}".format(all_obs.shape[0]))


    ### OBS FIGURES
    scatter(x = all_obs[:,0],
            y = all_obs[:,1],
            xlabel = "Vehicle Steer Angle",
            ylabel = "Vehicle Speed",
            title = "Dataset: Vehicle Steer vs. Vehicle Speed",
            log_dir = args.log_dir,
            f_name = "obs_scatter.png")

    hist(x = all_obs[:,0],
            bins = 15,
            xlabel = "Vehicle Steer Angle",
            ylabel = "Count",
            title = "Dataset: Steer Angle Histogram",
            log_dir = args.log_dir,
            f_name = "obs_steer_hist.png")

    hist(x = all_obs[:,1],
            bins = 15,
            xlabel = "Vehicle Speed",
            ylabel = "Count",
            title = "Dataset: Speed Histogram",
            log_dir = args.log_dir,
            f_name = "obs_speed_hist.png")

    ### ACTION FIGURES
    scatter(x = all_actions[:,0],
            y = all_actions[:,1],
            xlabel = "Action Steer",
            ylabel = "Action Speed",
            title = "Dataset: Action Steer vs. Action Speed",
            log_dir = args.log_dir,
            f_name = "action_scatter.png")

    hist(x = all_actions[:,0],
            bins = 15,
            xlabel = "Action Steer",
            ylabel = "Count",
            title = "Dataset: Action Steer Histogram",
            log_dir = args.log_dir,
            f_name = "action_steer_hist.png")

    hist(x = all_actions[:,1],
            bins = 15,
            xlabel = "Action Speed",
            ylabel = "Count",
            title = "Dataset: Action Speed Histogram",
            log_dir = args.log_dir,
            f_name = "action_speed_hist.png")

    ### DELTA FIGURES
    scatter(x = all_delta[:,1],
            y = all_delta[:,0],
            xlabel = "Delta Y",
            ylabel = "Delta X",
            title = "Dataset: Change in X vs. Change in Y",
            log_dir = args.log_dir,
            f_name = "delta_pos_scatter.png")

    hist(x = all_delta[:,0],
            bins = 15,
            xlabel = "Delta X",
            ylabel = "Count",
            title = "Dataset: Delta X Histogram",
            log_dir = args.log_dir,
            f_name = "delta_x_hist.png")

    hist(x = all_delta[:,1],
            bins = 15,
            xlabel = "Delta Y",
            ylabel = "Count",
            title = "Dataset: Delta Y Histogram",
            log_dir = args.log_dir,
            f_name = "delta_y_hist.png")

    hist(x = all_delta[:,2],
            bins = 15,
            xlabel = "Delta Theta",
            ylabel = "Count",
            title = "Dataset: Delta Theta Histogram",
            log_dir = args.log_dir,
            f_name = "delta_theta_hist.png")

    hist(x = all_delta[:,3],
            bins = 15,
            xlabel = "Delta Steer",
            ylabel = "Count",
            title = "Dataset: Delta Steeer Histogram",
            log_dir = args.log_dir,
            f_name = "delta_steer_hist.png")

    hist(x = all_delta[:,4],
            bins = 15,
            xlabel = "Delta Speed",
            ylabel = "Count",
            title = "Dataset: Delta Speed Histogram",
            log_dir = args.log_dir,
            f_name = "delta_speed_hist.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type = str)
    args = parser.parse_args()
    main(args)

