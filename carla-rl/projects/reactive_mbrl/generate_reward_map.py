import click
import hydra
import torch
import os
import glob
import numpy as np

from projects.reactive_mbrl.algorithms.multiprocessing import Multiprocessor
from projects.reactive_mbrl.algorithms.dynamic_programming import QSolver
from projects.reactive_mbrl.ego_model import EgoModel, EgoModelRails


def qsolver_worker(id, queue, model):
    try:
        while True:
            data_path = queue.get(timeout=60)
            data = load_data(data_path)
            solver = QSolver(model)
            solver.solve(data)
    except:
        print(id)
        raise


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    model = EgoModel()
    model.load_state_dict(model_weights)
    return model


def load_trajectory(path):
    reward_paths = sorted(
        [r for r in glob.glob("{}/reward/*.npy".format(path)) if "value" not in r]
    )[::-1]
    world_paths = sorted(glob.glob("{}/world/*.npy".format(path)))[::-1]
    assert len(reward_paths) == len(
        world_paths
    ), "Uneven number of reward/world/rgb paths"

    create_output_dir_if_not_exists(path, "value")
    create_output_dir_if_not_exists(path, "action_value")

    for reward_path, world_path in zip(reward_paths, world_paths):

        reward = torch.FloatTensor(np.load(reward_path))
        # reward = (preprocess_rgb(cv2.imread(reward_path), image_size=(16,16)) * 255)[0]
        # reward = (reward == 0).float()-1
        world_pts = torch.FloatTensor(np.load(world_path))
        value_path, action_value_path = construct_QV_paths(reward_path)

        yield reward, world_pts, value_path, action_value_path


def create_output_dir_if_not_exists(path, dir_name):
    output_dir = os.path.join(path, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def construct_QV_paths(reward_path):
    value_path = reward_path.split("/")
    value_path[-2] = "value"
    value_path = "/".join(value_path)

    action_value_path = reward_path.split("/")
    action_value_path[-2] = "action_value"
    action_value_path = "/".join(action_value_path)

    return value_path, action_value_path


@hydra.main(config_path="configs", config_name="reward_map.yaml")
def main(cfg):

    model = load_ego_model()
    trajectory_paths = glob.glob(cfg["dataset_path"] + "/*")

    if cfg["num_processes"] == 1:
        # Running single processing.
        solver = QSolver(model)
        for path in trajectory_paths:
            data = load_trajectory(path)
            solver.solve(data)
    else:
        processor = Multiprocessor(model, queue_size=len(trajectory_paths))
        processor.process(
            qsolver_worker, trajectory_paths, num_processes=int(cfg["num_processes"])
        )


if __name__ == "__main__":
    main()
