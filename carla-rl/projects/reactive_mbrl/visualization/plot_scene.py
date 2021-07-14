import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import hydra

from carla.libcarla import Vector3D

from projects.reactive_mbrl.create_env import create_env
from projects.reactive_mbrl.ego_model import EgoModel
from projects.reactive_mbrl.algorithms.dynamic_programming import QSolver
import projects.reactive_mbrl.data.reward_map as reward
import projects.reactive_mbrl.geometry.transformation as transform
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import LowDimObservationConfig

@hydra.main(config_path="../configs", config_name="config.yaml")
def run(config):
    output_path = config.eval['log_dir']
    env = create_env(config.env, output_path)
    env.reset()
    try:
        #plot_scene(env, output_path)
        plot_reward(env, output_path)
    except:
        env.close()
        raise
    finally:
        env.close()


def plot_reward(env, output_path):
    ax = plt.gca()
    rewards, _ = reward.calculate_reward_map(env)
    ax.pcolormesh(rewards)
    plt.savefig(os.path.join(output_path, 'fig.png'))


def plot_scene(env, output_path):
    ax = plt.gca()
    av_id = env.carla_interface.get_ego_vehicle()._vehicle.id

    plot_av(ax, env)
    plot_path(ax, env)
    plot_grid_points(ax, env)
    plot_actors(ax, env, include_ego=False, ego_id=av_id)
    plt.axis('equal')
    plt.savefig(os.path.join(output_path, 'fig.png'))

def plot_av(ax, env):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    plot_actor(ax, ego_actor, color='red')

def plot_grid_points(ax, env):
    world_points = reward.calculate_world_grid_points(env)
    rewards = reward.calculate_path_following_reward(env, world_points)
    #rewards, world_points = reward.calculate_reward_map(env, {})
    lower, upper = min(rewards), max(rewards)
    print(f"rewards is {rewards}")
    rewards_norm = (rewards - lower) / (upper - lower)
    print(f"rewards norm is {rewards_norm}")
    for (pt, r) in zip(world_points, rewards_norm):
        alpha = r if r > 0.1 else 0.1
        ax.plot(pt[0], pt[1], 'o', color='black', alpha=alpha)

def plot_actors(ax, env, include_ego=False, ego_id=0):
    for actor in env.carla_interface.actor_fleet.actor_list:
        if actor.type_id != 'vehicle':
            continue

        if not include_ego and actor.id == ego_id:
            continue

        plot_actor(ax, actor, color='black')

def plot_actor(ax, actor, color='black'):
    bounding_box = get_local_points(actor.bounding_box.extent)
    actor_global = transform.transform_points(actor.get_transform(), bounding_box)
    plt.plot(actor_global[:, 0], actor_global[:, 1], color=color)

def plot_path(ax, env):
    path = env.carla_interface.next_waypoints
    path_points = np.array([waypoint_to_numpy(wpt) for wpt in path])
    ax.plot(path_points[:, 0], path_points[:, 1], color='blue')

def waypoint_to_numpy(waypoint):
    return [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z]

def get_local_points(extent):
    return np.array([
        [-extent.x, extent.y, 0, 1],
        [extent.x, extent.y, 0, 1],
        [extent.x, -extent.y, 0, 1],
        [-extent.x, -extent.y, 0, 1]])


if __name__ == "__main__":
    run()
