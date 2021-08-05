import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import hydra
import copy

from carla.libcarla import Vector3D

from projects.reactive_mbrl.create_env import create_env
from projects.reactive_mbrl.ego_model import EgoModel
from projects.reactive_mbrl.algorithms.dynamic_programming import QSolver
import projects.reactive_mbrl.data.reward_map as reward
import projects.reactive_mbrl.geometry.transformation as transform
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import LowDimObservationConfig

from projects.reactive_mbrl.algorithms.dynamic_programming import ACTIONS, num_acts
from projects.reactive_mbrl.algorithms.dynamic_programming import SPEEDS, YAWS
from projects.reactive_mbrl.data.reward_map import MAP_SIZE

ACTIONS1 = np.array([[-1, 1 / 3], [-1, 2 / 3], [-1, 1],], dtype=np.float32,).reshape(
    3, 2
)

ACTIONS2 = np.array([[0, 1 / 3], [0, 2 / 3], [0, 1],], dtype=np.float32,).reshape(3, 2)

ACTIONS3 = np.array([[1, 1 / 3], [1, 2 / 3], [1, 1],], dtype=np.float32,).reshape(3, 2)

num_acts = len(ACTIONS1)
num_yaws = len(YAWS)
num_spds = len(SPEEDS)


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    model = EgoModel(dt=10.0)
    model.load_state_dict(model_weights)
    return model


@hydra.main(config_path="../configs", config_name="config.yaml")
def run(config):
    output_path = os.path.join(config.eval["log_dir"], "raw_maps")
    env = create_env(config.env, output_path)
    env.reset(unseen=False, index=0)
    waypoints = env.carla_interface.global_planner._waypoints_queue
    waypoints = np.array(
        [
            [
                w[0].transform.location.x,
                w[0].transform.location.y,
                w[0].transform.rotation.yaw,
            ]
            for w in waypoints
        ]
    )
    try:
        # Warm start
        # ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        # ego_loc = ego_actor.get_transform().location
        for _ in range(100):
            expert_action = env.get_autopilot_action(target_speed=10.0)
            _, _, _, info = env.step(expert_action)
        for idx in range(0, 1000):
            _, world_points, _ = reward.calculate_reward_map(env, waypoints)
            expert_action = env.get_autopilot_action(target_speed=10.0)
            _, _, _, info = env.step(expert_action)
            if idx % 20 == 0:
                # ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
                # ego_loc = ego_actor.get_transform().location
                # print(f"acter 30 steps {ego_loc.x}, {ego_loc.y}")
                # _, world_points, _ = reward.calculate_reward_map(env, waypoints)
                # plot_reward(env, output_path, waypoints, idx)
                # expert_action = env.get_autopilot_action(target_speed=5.0)
                # env.step(expert_action)
                _, next_world_points, _ = reward.calculate_reward_map(env, waypoints)
                plot_world_and_next(
                    env, world_points, next_world_points, output_path, idx
                )
                # plot_world(env, info, world_points, output_path, idx)

        # plot_scene(env, output_path)
        # plot_route(env, waypoints, output_path)
        # plot_reward(env, output_path, waypoints)
    except:
        env.close()
        raise
    finally:
        env.close()


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    model = EgoModel()
    model.load_state_dict(model_weights)
    return model


def get_closest_waypoint(loc, route):
    min_dist = 10000
    min_wpt = None
    for (wpt_x, wpt_y, wpt_yaw) in route:
        d = np.linalg.norm(loc - np.array([wpt_x, wpt_y]))
        if d < min_dist:
            min_dist = d
            min_wpt = (np.array([wpt_x, wpt_y]), wpt_yaw)
    return min_wpt


def plot_world(env, info, world_points, output_path, idx):
    route = env.carla_interface.next_waypoints
    route = np.array(
        [
            [
                w.transform.location.x,
                w.transform.location.y,
                np.radians(w.transform.rotation.yaw),
            ]
            for w in route
        ]
    )
    model = load_ego_model()

    num_pts, _ = world_points.shape
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    ego_loc = ego_actor.get_transform().location
    base_transform = ego_actor.get_transform()

    locs = np.array([base_transform.location.x, base_transform.location.y])
    closest_waypoint = get_closest_waypoint(locs, route)
    locs = np.tile(locs, (num_acts, 1))

    ego_yaw = np.radians(base_transform.rotation.yaw)
    yaws = np.array([ego_yaw])
    yaws = np.tile(yaws, (num_acts, 1))

    print(f"Ego speed is {info['speed']}")
    speeds = np.array([info["speed"]])
    speeds = np.tile(speeds, (num_acts, 1))

    locs = torch.tensor(locs)
    yaws = torch.tensor(yaws)
    speeds = torch.tensor(speeds)
    actions = torch.tensor(ACTIONS)

    pred_locs, pred_yaws, pred_speeds = model.forward(locs, yaws, speeds, actions)
    pred_locs = pred_locs.detach().numpy()
    pred_yaws = pred_yaws.detach().numpy()
    pred_speeds = pred_speeds.detach().numpy()
    loc_loss, yaw_loss, speed_loss, action_value = reward.calculate_action_value_map(
        pred_locs, pred_yaws, pred_speeds, closest_waypoint
    )
    action_value = action_value[:27].reshape(9, 3)
    loc_loss = loc_loss[:27].reshape(9, 3)
    yaw_loss = yaw_loss[:27].reshape(9, 3)
    speed_loss = speed_loss[:27].reshape(9, 3)

    topdown = info["sensor.camera.rgb/top"]

    fig, axes = plt.subplots(1, 7, figsize=(70, 10))

    ax = axes[0]
    ego_speed = info["speed"]
    ax.annotate(f"ego_speed = {ego_speed}, ego_yaw = {ego_yaw}", xy=(0, 0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax = axes[1]
    ax.imshow(topdown)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax = axes[2]
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    ego_loc = ego_actor.get_transform().location
    plot_actor(ax, ego_actor, color="red")
    ax.plot(route[:, 0], route[:, 1], "o", color="blue")
    ax.plot(closest_waypoint[0][0], closest_waypoint[0][1], "o", color="green")
    ax.plot(pred_locs[:, 0], pred_locs[:, 1], "o", color="red")
    ax.set_xlim(ego_loc.x - 20, ego_loc.x + 20)
    ax.set_ylim(ego_loc.y - 20, ego_loc.y + 20)

    ax = axes[3]
    im = ax.pcolormesh(action_value)
    fig.colorbar(im, ax=ax)
    ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Total loss")

    ax = axes[4]
    im = ax.pcolormesh(loc_loss)
    fig.colorbar(im, ax=ax)
    ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Loc loss")

    ax = axes[5]
    im = ax.pcolormesh(yaw_loss)
    fig.colorbar(im, ax=ax)
    ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Yaw loss")

    ax = axes[6]
    im = ax.pcolormesh(speed_loss)
    fig.colorbar(im, ax=ax)
    ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Speed loss")

    plt.savefig(os.path.join(output_path, f"reward_{idx}.png"))


def plot_route(env, route, output_path):
    ax = plt.gca()
    ax.plot(route[:, 0], route[:, 1], color="blue")
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    ego_loc = ego_actor.get_transform().location
    plot_actor(ax, ego_actor, color="red")
    ax.set_xlim(ego_loc.x - 10, ego_loc.x + 10)
    ax.set_ylim(ego_loc.y - 10, ego_loc.y + 10)
    plt.savefig(os.path.join(output_path, "route.png"))


def plot_world_and_next(env, world_pts, next_world_pts, output_path, index):
    ax = plt.gca()
    predicted = predict_next_world_pts(env, world_pts, torch.tensor(ACTIONS3))

    # offset = next_world_pts[0]
    # world_pts -= offset
    # predicted -= offset
    # next_world_pts -= offset

    ax.plot(
        world_pts[:, 0], world_pts[:, 1], "o", color="black", label="previous",
    )
    ax.plot(
        next_world_pts[:, 0], next_world_pts[:, 1], "o", color="green", label="real",
    )

    plot_av(ax, env)
    plot_path(ax, env)
    ax.plot(
        predicted[:, 0],
        predicted[:, 1],
        "o",
        color="red",
        label="predicted 1 steering",
    )

    predicted = predict_next_world_pts(env, world_pts, torch.tensor(ACTIONS1))
    ax.plot(
        predicted[:, 0],
        predicted[:, 1],
        "o",
        color="red",
        label="predicted -1 steering",
    )

    predicted = predict_next_world_pts(env, world_pts, torch.tensor(ACTIONS2))
    ax.plot(
        predicted[:, 0],
        predicted[:, 1],
        "o",
        color="red",
        label="predicted 0 steering",
    )

    theta = np.arctan2(next_world_pts[-1][1], next_world_pts[-1][0])
    print(f"angle is {np.pi/4 - theta}")
    _next_locs = rotate_pts(next_world_pts, (np.pi / 4) - theta)
    # ax.plot(
    #    _next_locs[:, 0], _next_locs[:, 1], "o", color="red", label="predicted rotated",
    # )
    ax.legend()
    ax.axis("equal")
    plt.savefig(os.path.join(output_path, f"world_{index}.png"))
    plt.clf()


def plot_reward(env, output_path, waypoints, idx):
    ax = plt.gca()
    rewards, _, _ = reward.calculate_reward_map(env, waypoints)
    ax.pcolormesh(rewards)
    plt.savefig(os.path.join(output_path, f"reward_{idx}.png"))
    plt.clf()


def plot_scene(env, output_path):
    ax = plt.gca()
    av_id = env.carla_interface.get_ego_vehicle()._vehicle.id

    plot_av(ax, env)
    plot_path(ax, env)
    plot_grid_points(ax, env)
    plot_actors(ax, env, include_ego=False, ego_id=av_id)
    # plot_predicted_positions(ax, env)
    plt.axis("equal")
    plt.savefig(os.path.join(output_path, "fig.png"))


def plot_av(ax, env):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    plot_actor(ax, ego_actor, color="red")


def plot_grid_points(ax, env):
    # world_points = reward.calculate_world_grid_points(env)
    # rewards = reward.calculate_path_following_reward(env, world_points)
    rewards, world_points, pixel_xy = reward.calculate_reward_map(env)
    lower, upper = rewards.min(), rewards.max()
    print(f"rewards is {rewards}")
    rewards_norm = (rewards - lower) / (upper - lower)
    print(f"rewards norm is {rewards_norm}")
    for (pt, xy) in zip(world_points, pixel_xy):
        r = rewards_norm[int(xy[0]), int(xy[1])]
        alpha = r if r > 0.1 else 0.1
        ax.plot(pt[0], pt[1], "o", color="black", alpha=alpha)
        if xy[0] == 0 and xy[1] == 12:
            ax.plot(pt[0], pt[1], "o", color="red", alpha=alpha)

    next_world_points = predict_next_world_pts(
        env, world_points, torch.tensor(ACTIONS1)
    )
    # ax.plot(
    #     next_world_points[:, 0],
    #     next_world_points[:, 1],
    #     "o",
    #     color="red",
    #     label="-1 steer",
    # )
    # next_world_points = predict_next_world_pts(
    #     env, world_points, torch.tensor(ACTIONS2)
    # )
    # ax.plot(
    #     next_world_points[:, 0],
    #     next_world_points[:, 1],
    #     "o",
    #     color="blue",
    #     label="0 steer",
    # )
    # next_world_points = predict_next_world_pts(
    #     env, world_points, torch.tensor(ACTIONS3)
    # )
    # ax.plot(
    #     next_world_points[:, 0],
    #     next_world_points[:, 1],
    #     "o",
    #     color="green",
    #     label="1 steer",
    # )
    ax.legend()


def predict_next_world_pts(env, world_points, actions):
    model = load_ego_model()
    locs = torch.tensor(world_points)
    speed = torch.tensor(5.0)
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    base_transform = ego_actor.get_transform()
    yaw = torch.tensor(np.radians(base_transform.rotation.yaw))
    pred_locs, pred_yaws, pred_spds = model.forward(
        locs[:, None, :].repeat(1, num_acts, 1).reshape(-1, 2),
        yaw[None, None].repeat(MAP_SIZE * MAP_SIZE, num_acts, 1).reshape(-1, 1),
        speed[None, None].repeat(MAP_SIZE * MAP_SIZE, num_acts, 1).reshape(-1, 1),
        actions[None].repeat(MAP_SIZE * MAP_SIZE, 1, 1).reshape(-1, 2),
    )

    return pred_locs.detach().numpy()


def plot_actors(ax, env, include_ego=False, ego_id=0):
    for actor in env.carla_interface.actor_fleet.actor_list:
        if actor.type_id != "vehicle":
            continue

        if not include_ego and actor.id == ego_id:
            continue

        plot_actor(ax, actor, color="black")


def plot_actor(ax, actor, color="black"):
    bounding_box = get_local_points(actor.bounding_box.extent)
    actor_global = transform.transform_points(actor.get_transform(), bounding_box)
    ax.plot(actor_global[:, 0], actor_global[:, 1], color=color)


def plot_path(ax, env):
    path = env.carla_interface.next_waypoints
    path_points = np.array([waypoint_to_numpy(wpt) for wpt in path])
    ax.plot(path_points[:, 0], path_points[:, 1], color="blue")


def waypoint_to_numpy(waypoint):
    return [
        waypoint.transform.location.x,
        waypoint.transform.location.y,
        waypoint.transform.location.z,
    ]


def get_local_points(extent):
    return np.array(
        [
            [-extent.x, extent.y, 0, 1],
            [extent.x, extent.y, 0, 1],
            [extent.x, -extent.y, 0, 1],
            [-extent.x, -extent.y, 0, 1],
        ]
    )


def rotate_pts(pts, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(pts.T).T


if __name__ == "__main__":
    run()
