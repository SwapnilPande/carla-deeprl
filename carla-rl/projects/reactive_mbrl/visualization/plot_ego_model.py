import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import os

from environment import CarlaEnv
from projects.reactive_mbrl.ego_model import EgoModel, EgoModelRails
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *


@click.command()
@click.option('--output-path', type=str, required=True)
def run(output_path: str):
    env = create_env(output_path)
    env.reset()
    model_weights = torch.load(os.path.join(os.getcwd(), 'carla-rl/projects/reactive_mbrl/ego_model.th'))
    model = EgoModel()
    #model_weights = torch.load(os.path.join(os.getcwd(), 'carla-rl/projects/reactive_mbrl/ego_model_rails.th'))
    #model = EgoModelRails(dt=1./4)
    model.load_state_dict(model_weights)
    try:
        plot_ego_model(env, model, output_path)
    except:
        env.close()
        raise
    finally:
        env.close()


def create_env(log_dir):
    config = DefaultMainConfig()
    config.server_fps = 20

    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/top'] = {
        'x':13.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'64',
        'sensor_y_res':'64',
        'fov':'90', \
        'sensor_tick': '0.0'}
    obs_config.sensors['sensor.camera.rgb/map'] = {
        'x':13.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'16',
        'sensor_y_res':'16',
        'fov':'90', \
        'sensor_tick': '0.0'}

    scenario_config = NoCrashDenseTown01Config()

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(observation_config=obs_config, scenario_config=scenario_config)
    env = CarlaEnv(config=config, log_dir=log_dir + '/')
    return env


def plot_ego_model(env, model, output_path):
    obs = env.reset()

    # Warm up.
    for i in range(101):
        expert_action = env.get_autopilot_action(.5)
        next_obs, reward, done, info = env.step(expert_action)

    real_pos = [extract_av_pos(env)]
    pred_pos = [extract_av_pos(env)]
    cur_loc = real_pos[0]
    for i in range(10):
        expert_action = env.get_autopilot_action(.5)
        next_obs, reward, done, info = env.step(expert_action)
        cur_loc = get_pred_pos(env, cur_loc, info, model, expert_action)
        pred_pos.append(cur_loc)
        real_pos.append(extract_av_pos(env))
    real_pos = np.array(real_pos)
    print(real_pos)
    for idx, (pos1, pos2) in enumerate(zip(real_pos, pred_pos)):
        if idx == 0:
            plt.plot(pos1[0], pos1[1], 'o', color='red')
            plt.plot(pos2[0], pos2[1], 'o', color='red')
        else:
            plt.plot(pos1[0], pos1[1], 'o', color='blue')
            plt.plot(pos2[0], pos2[1], 'o', color='green')
    plt.axis('equal')
    plt.savefig(os.path.join(output_path, 'ego.png'))


def get_pred_pos(env, loc, info, model, expert_action):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    ego_transform = ego_actor.get_transform()
    #loc = [ego_transform.location.x, ego_transform.location.y]
    expert_action[1] = 0
    yaw = [ego_transform.rotation.yaw * math.pi / 180.0]
    speed = info['speed']

    next_loc, _, _ = model.forward(
        torch.tensor(np.expand_dims(loc, axis=0)).float(),
        torch.tensor(np.expand_dims(yaw, axis=0)).float(),
        torch.tensor(np.expand_dims(speed, axis=0)).float(),
        torch.tensor(np.expand_dims(expert_action, axis=0)).float())
    return next_loc[0].detach().numpy()


def extract_av_pos(env):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    matrix = np.array(ego_actor.get_transform().get_matrix())
    return matrix[:2, 3]


if __name__ == "__main__":
    run()
