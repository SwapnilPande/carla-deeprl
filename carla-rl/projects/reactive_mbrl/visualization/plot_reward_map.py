import click
import glob
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json

from projects.reactive_mbrl.ego_model import EgoModel, EgoModelRails
from projects.reactive_mbrl.data.reward_map import MAP_SIZE
from projects.reactive_mbrl.algorithms.dynamic_programming import ACTIONS, SPEEDS, YAWS

SEM_COLORS = {
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    10: (0, 0, 142),
    18: (220, 220, 0),
}

num_acts = len(ACTIONS)
num_yaws = len(YAWS)
num_spds = len(SPEEDS)


def convert_sem(sem, labels=[4, 6, 7, 10, 18]):
    canvas = np.zeros(sem.shape + (3,), dtype=np.uint8)
    for i, label in enumerate(labels):
        canvas[sem == i + 1] = SEM_COLORS[label]

    return canvas


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    model = EgoModel()
    model.load_state_dict(model_weights)
    return model


def predict_next_locs(model):
    speeds, yaws, actions = (
        torch.tensor(SPEEDS),
        torch.tensor(YAWS),
        torch.tensor(ACTIONS),
    )
    locs = torch.tensor([[0.0, 0.0]])
    speed = torch.tensor(5.0)
    yaw = torch.tensor(0.0)
    pred_locs, pred_yaws, pred_spds = model.forward(
        locs[:, None, :].repeat(1, num_acts, 1).reshape(-1, 2),
        yaw[None, None].repeat(1, num_acts, 1).reshape(-1, 1),
        speed[None, None].repeat(1, num_acts, 1).reshape(-1, 1),
        actions[None].repeat(1, 1, 1).reshape(-1, 2),
    )

    import pdb

    pdb.set_trace()


@click.command()
@click.option("--dataset-path", type=str, required=True)
@click.option("--output-path", type=str, required=True)
def plot_reward_map(dataset_path, output_path):
    model = load_ego_model()
    # next_locs = predict_next_locs(model)

    trajectory_paths = glob.glob(f"{dataset_path}/*")
    trajectory_path = trajectory_paths[0]
    paths = glob.glob(f"{trajectory_path}/narrow_rgb/*.png")

    # for idx in range(1000, 2000, 10):
    for idx in range(0, len(paths), 10):

        narr_rgb_path = sorted(glob.glob(f"{trajectory_path}/narrow_rgb/*.png"))[idx]
        narr_rgb = cv2.imread(narr_rgb_path)

        narr_seg_path = sorted(glob.glob(f"{trajectory_path}/narrow_seg/*.png"))[idx]
        narr_seg = cv2.imread(narr_seg_path, cv2.IMREAD_GRAYSCALE)
        narr_seg = convert_sem(narr_seg)

        wide_rgb_path = sorted(glob.glob(f"{trajectory_path}/wide_rgb/*.png"))[idx]
        wide_rgb = cv2.imread(wide_rgb_path)

        wide_seg_path = sorted(glob.glob(f"{trajectory_path}/wide_seg/*.png"))[idx]
        wide_seg = cv2.imread(wide_seg_path, cv2.IMREAD_GRAYSCALE)
        wide_seg = convert_sem(wide_seg)

        reward_path = sorted(glob.glob(f"{trajectory_path}/reward/*.npy"))[idx]
        reward = np.load(reward_path)

        top_rgb_path = sorted(glob.glob(f"{trajectory_path}/topdown/*.png"))[idx]
        top_rgb = cv2.imread(top_rgb_path)

        value_path = sorted(glob.glob(f"{trajectory_path}/value/*.npy"))[idx]
        value = np.load(value_path)

        world_path = sorted(glob.glob(f"{trajectory_path}/world/*.npy"))[idx]
        world_pts = np.load(world_path)

        action_value_path = sorted(glob.glob(f"{trajectory_path}/action_value/*.npy"))[
            idx
        ]
        action_value = np.load(action_value_path)

        json_path = sorted(glob.glob(f"{trajectory_path}/measurements/*.json"))[idx]
        with open(json_path, "r") as f:
            measurement = json.load(f)
        ego_pos = np.array([measurement["ego_vehicle_x"], measurement["ego_vehicle_y"]])

        action_value = action_value[int(MAP_SIZE / 2), int(MAP_SIZE / 2), 3, 2]
        action_value = action_value[:27].reshape(9, 3)

        pixel_x, pixel_y = np.meshgrid(np.arange(MAP_SIZE), np.arange(MAP_SIZE))
        pixel_xy = np.stack([pixel_x.flatten(), pixel_y.flatten()], axis=-1)

        def distance(pt1, pt2):
            return np.linalg.norm(pt1 - pt2)

        index = np.argsort([distance(pt, ego_pos) for pt in world_pts])[0]
        closest_point = pixel_xy[index]
        print(f"{closest_point[0]}, {closest_point[1]}")

        # plot_grid_points(
        #     world_pts, world_pts[index], ego_pos, output_path, idx,
        # )

        print(reward.max())

        plot_value_map(
            narr_rgb,
            wide_rgb,
            narr_seg,
            wide_seg,
            top_rgb,
            value,
            action_value,
            output_path,
            idx,
        )


def plot_grid_points(world_pts, closest_point, ego_pos, output_path, index):
    plt.plot(world_pts[:, 0], world_pts[:, 1], "o", color="black")
    plt.plot(ego_pos[0], ego_pos[1], "x", color="red")
    plt.plot(closest_point[0], closest_point[1], color="blue")
    plt.axis("equal")
    plt.savefig(os.path.join(output_path, f"value_map_{index}.png"))
    plt.clf()


def plot_value_map(
    narr_rgb,
    wide_rgb,
    narr_seg,
    wide_seg,
    top_rgb,
    value,
    action_value,
    output_path,
    idx,
):
    fig = plt.figure(constrained_layout=True, figsize=((25, 10)))
    gs = fig.add_gridspec(nrows=4, ncols=17)

    for s, speed, in enumerate(SPEEDS):
        for y, yaw in enumerate(YAWS):
            m = value[:, :, s, y]
            ax = fig.add_subplot(gs[s, 2 * y : 2 * (y + 1)])
            ax.pcolormesh(m)
            ax.set_title(f"speed={speed:2.2f}, yaw={yaw:2.2f}", fontsize=7)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    ax = fig.add_subplot(gs[0, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(wide_rgb)

    ax = fig.add_subplot(gs[1, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(wide_seg)

    ax = fig.add_subplot(gs[2, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(narr_rgb)

    ax = fig.add_subplot(gs[3, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(narr_seg)

    ax = fig.add_subplot(gs[0:2, 15:])
    im = ax.pcolormesh(action_value)
    fig.colorbar(im, ax=ax)
    ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    ax = fig.add_subplot(gs[2:, 15:])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(top_rgb)

    # fig.tight_layout()
    img_path = os.path.join(f"{output_path}/value_map_{idx}.png")
    plt.savefig(img_path)
    plt.clf()


if __name__ == "__main__":
    plot_reward_map()
