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
from projects.reactive_mbrl.algorithms.dynamic_programming import (
    ACTIONS,
    SPEEDS,
    YAWS,
    NORMALIZING_ANGLE,
    initialize_grid_interpolator,
    interpolate,
)

ACTIONS1 = np.array([[-1, 1 / 3], [0, 1 / 3], [1, 1 / 3],], dtype=np.float32,).reshape(
    3, 2
)

speeds, yaws, actions = (
    torch.tensor(SPEEDS),
    torch.tensor(YAWS),
    torch.tensor(ACTIONS),
)
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


def predict_next_locs(locs, yaw, speed, model):
    speeds, yaws, actions = (
        torch.tensor(SPEEDS),
        torch.tensor(YAWS),
        torch.tensor(ACTIONS1),
    )
    num_acts = len(ACTIONS1)

    pred_locs, pred_yaws, pred_spds = model.forward(
        locs[:, None, :].repeat(1, num_acts, 1).reshape(-1, 2),
        yaw[None, None].repeat(1, num_acts, 1).reshape(-1, 1),
        speed[None, None].repeat(1, num_acts, 1).reshape(-1, 1),
        actions[None].repeat(1, 1, 1).reshape(-1, 2),
    )
    return pred_locs, pred_yaws, pred_spds


@click.command()
@click.option("--dataset-path", type=str, required=True)
@click.option("--output-path", type=str, required=True)
def plot_reward_map(dataset_path, output_path):
    model = load_ego_model()
    # next_locs = predict_next_locs(model)

    trajectory_paths = glob.glob(f"{dataset_path}/*")
    trajectory_path = trajectory_paths[1]
    paths = glob.glob(f"{trajectory_path}/narrow_rgb/*.png")
    next_locs = None
    prev_V = None

    # for idx in range(1000, 2000, 10):
    for idx in range(0, len(paths), 20):
        # for idx in range(len(paths) - 1, len(paths) - 500, -1):

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

        next_path = sorted(glob.glob(f"{trajectory_path}/next_preds/*.npy"))[idx]
        next_preds = np.load(next_path)

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

        # repeats = np.ones_like(action_value, dtype=int)
        # repeats[-1] = 3
        # Make this 10 x 3 to include the brake action
        # action_value = np.repeat(action_value, repeats)
        # action_value = action_value.reshape(10, 3)
        action_value = action_value.reshape(9, 3)

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

        if next_locs is None:
            next_locs = world_pts

        # if prev_V is None:
        #     prev_V = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1).repeat(
        #         1, 1, num_spds, num_yaws
        #     )

        # grid_interpolator, theta, offset = initialize_grid_interpolator(
        #     next_locs, prev_V, speeds, yaws
        # )

        print(reward.max())
        # plot_world_and_next(
        #     ego_pos, top_rgb, reward, next_preds, world_pts, output_path, idx
        # )
        next_locs = world_pts
        # plot_action_value_from_value(
        #     reward, top_rgb, world_pts, measurement, output_path, idx
        # )

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


def plot_action_value_from_value(
    reward, topdown, world_points, measurement, output_path, idx
):
    reward = reward[::-1]
    reward = np.copy(reward)
    reward = torch.tensor(reward)
    reward = reward.reshape(MAP_SIZE, MAP_SIZE)
    value = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1).repeat(1, 1, num_spds, num_yaws)
    value = value.detach().numpy()
    pixel_x, pixel_y = np.meshgrid(np.arange(MAP_SIZE), np.arange(MAP_SIZE))
    pixel_xy = np.stack(
        [pixel_x.flatten(), pixel_y.flatten(), np.ones(MAP_SIZE * MAP_SIZE)], axis=-1
    )

    grid_interpolator, theta, offset = initialize_grid_interpolator(
        world_points, value, speeds, yaws
    )
    ego_yaw = np.radians(measurement["yaw"])
    ego_pos = np.array([measurement["ego_vehicle_x"], measurement["ego_vehicle_y"]])
    model = load_ego_model()
    pred_locs, pred_yaws, pred_speeds = predict_next_locs(
        torch.tensor([ego_pos]), torch.tensor(ego_yaw), torch.tensor(5.0), model
    )
    pred_locs = pred_locs.detach().numpy()
    pred_yaws = pred_yaws.detach().numpy()
    pred_speeds = pred_speeds.detach().numpy()
    next_Vs = interpolate(
        grid_interpolator, pred_locs, pred_speeds, pred_yaws - ego_yaw, offset, theta,
    )

    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    ax = axes[0]
    ax.imshow(topdown)

    ax = axes[1]
    ax.plot(ego_pos[0], ego_pos[1], "x", color="red")
    ax.plot(pred_locs[0, 0], pred_locs[0, 1], "o", color="green", label="-1 steering")
    ax.plot(pred_locs[1, 0], pred_locs[1, 1], "o", color="green", label="0 steering")
    ax.plot(pred_locs[2, 0], pred_locs[2, 1], "o", color="green", label="1 steering")
    for (pixels, world_pt) in zip(pixel_xy, world_points):
        px = pixels[0]
        py = pixels[1]
        r = value[int(px), int(py), 3, 2] / 3
        ax.plot(
            world_pt[0], world_pt[1], "o", color="black", alpha=min(r + 1.1, 1),
        )
    ax.axis("equal")
    ax.legend()

    ax = axes[2]
    rotated_world_points = normalize_points(world_points, offset, theta)
    rotated_pred_locs = normalize_points(pred_locs, offset, theta)
    rotated_ego_pos = normalize_points(ego_pos, offset, theta)
    ax.plot(rotated_ego_pos[0], rotated_ego_pos[1], "x", color="red")
    for (pixels, world_pt) in zip(pixel_xy, rotated_world_points):
        px = pixels[0]
        py = pixels[1]
        r = value[int(px), int(py), 3, 2] / 3
        ax.plot(
            world_pt[0], world_pt[1], "o", color="black", alpha=min(r + 1.1, 1),
        )

    # ax.plot(rotated_world_points[:, 0], rotated_world_points[:, 1], "o", color="black")
    ax.plot(
        rotated_pred_locs[0, 0],
        rotated_pred_locs[0, 1],
        "o",
        color="green",
        label="-1 steering",
    )
    ax.plot(
        rotated_pred_locs[1, 0],
        rotated_pred_locs[1, 1],
        "o",
        color="green",
        label="0 steering",
    )
    ax.plot(
        rotated_pred_locs[2, 0],
        rotated_pred_locs[2, 1],
        "o",
        color="green",
        label="1 steering",
    )
    ax.axis("equal")
    ax.legend()

    ax = axes[3]
    ax.pcolormesh(value[:, :, 3, 2])

    ax = axes[4]
    im = ax.pcolormesh(next_Vs[..., None])
    fig.colorbar(im, ax=ax)

    plt.savefig(os.path.join(output_path, f"world_{idx}.png"))
    plt.clf()


def rotate_and_offset(locs):
    offset = locs[0]
    locs = locs - offset
    theta = np.arctan2(locs[-1][1], locs[-1][0])
    locs = rotate_pts(locs, NORMALIZING_ANGLE - theta)
    return locs, theta, offset


def rotate_pts(pts, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(pts.T).T


def normalize_points(points, offset, theta):
    points = points - offset
    points = rotate_pts(points, NORMALIZING_ANGLE - theta)
    return points


def plot_world_and_next(
    ego_pos, topdown, reward, next_preds, world_pts, output_path, index
):
    plt.figure(figsize=(50, 10))
    fig, axes = plt.subplots(1, 4)

    ax = axes[0]
    ax.plot(world_pts[:, 0], world_pts[:, 1], "o", color="black")
    ax.plot(next_preds[:, 0], next_preds[:, 1], "o", color="blue")
    ax.plot(world_pts[-1, 0], world_pts[-1, 1], "o", color="green")
    ax.plot(next_preds[-1, 0], next_preds[-1, 1], "o", color="green")
    ax.plot(ego_pos[0], ego_pos[1], "x", color="red")
    ax.axis("equal")

    rotated_world_pts, theta, offset = rotate_and_offset(world_pts)
    rotated_next_preds = normalize_points(next_preds, offset, theta)
    rotated_ego_pos = normalize_points(ego_pos, offset, theta)

    ax = axes[1]
    ax.plot(rotated_world_pts[:, 0], rotated_world_pts[:, 1], "o", color="black")
    ax.plot(rotated_next_preds[:, 0], rotated_next_preds[:, 1], "o", color="blue")
    ax.plot(rotated_world_pts[-1, 0], rotated_world_pts[-1, 1], "o", color="green")
    ax.plot(rotated_next_preds[-1, 0], rotated_next_preds[-1, 1], "o", color="green")
    ax.plot(rotated_ego_pos[0], rotated_ego_pos[1], "x", color="red")
    ax.axis("equal")

    ax = axes[2]
    reward = np.moveaxis(reward, 0, 1)
    im = ax.pcolormesh(reward)
    fig.colorbar(im, ax=ax)
    ax.axis("equal")

    ax = axes[3]
    ax.imshow(topdown)
    ax.axis("equal")

    plt.savefig(os.path.join(output_path, f"value_map_{index}.png"))
    plt.clf()


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
