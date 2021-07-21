import hydra
import glob
import os
import torch
import numpy as np
import json

from projects.reactive_mbrl.algorithms.dynamic_programming import ACTIONS, num_acts
from projects.reactive_mbrl.ego_model import EgoModel, EgoModelRails


def create_output_dir_if_not_exists(path, dir_name):
    output_dir = os.path.join(path, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def load_trajectory(path):
    reward_paths = sorted(
        [r for r in glob.glob("{}/reward/*.npy".format(path)) if "value" not in r]
    )[::-1]
    world_paths = sorted(glob.glob("{}/world/*.npy".format(path)))[::-1]
    measurement_paths = sorted(glob.glob("{}/measurements/*.json".format(path)))[::-1]
    assert (
        len(reward_paths) == len(world_paths) == len(measurement_paths)
    ), "Uneven number of reward/world/rgb paths"

    create_output_dir_if_not_exists(path, "value")
    create_output_dir_if_not_exists(path, "action_value")

    for reward_path, world_path, measurement_path in zip(
        reward_paths, world_paths, measurement_paths
    ):
        # reward = np.load(reward_path)[::-1]
        # reward = torch.FloatTensor(np.copy(reward))
        reward = torch.FloatTensor(np.load(reward_path))
        # reward = (preprocess_rgb(cv2.imread(reward_path), image_size=(16,16)) * 255)[0]
        # reward = (reward == 0).float()-1
        world_pts = torch.FloatTensor(np.load(world_path))
        value_path, action_value_path = construct_QV_paths(reward_path)
        with open(measurement_path, "r") as f:
            measurement = json.load(f)

        yield reward, world_pts, value_path, action_value_path, measurement


def construct_QV_paths(reward_path):
    value_path = reward_path.split("/")
    value_path[-2] = "value"
    value_path = "/".join(value_path)

    action_value_path = reward_path.split("/")
    action_value_path[-2] = "action_value"
    action_value_path = "/".join(action_value_path)

    return value_path, action_value_path


def calculate_raw_map(model, data):
    for reward, world_pts, value_path, action_val_path, measurements in data:
        locs = np.array([measurements["ego_vehicle_x"], measurements["ego_vehicle_y"]])
        locs = np.tile(locs, (num_acts, 1))
        yaws = np.array([measurements["yaw"]])
        yaws = np.tile(yaws, (num_acts, 1))
        speeds = np.array([measurements["speed"]])
        speeds = np.tile(speeds, (num_acts, 1))

        locs = torch.tensor(locs)
        yaws = torch.tensor(yaws)
        speeds = torch.tensor(speeds)
        actions = torch.tensor(ACTIONS)

        pred_locs, pred_yaws, pred_speeds = model.forward(locs, yaws, speeds, actions)
        action_value = reward.calculate_action_value_map(pred_locs, pred_yaws, pred_speeds)

        np.save(action_value, action_val_path)
        # np.save(value, value_path)


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    model = EgoModel()
    model.load_state_dict(model_weights)
    return model


@hydra.main(config_path="configs", config_name="reward_map.yaml")
def main(cfg):
    trajectory_paths = glob.glob(cfg["dataset_path"] + "/*")
    model = load_ego_model()
    for path in trajectory_paths:
        data = load_trajectory(path)
        calculate_raw_map(model, data)


if __name__ == "__main__":
    main()
