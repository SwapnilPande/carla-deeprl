import os
import sys
import json
import datetime
import argparse
import random
import math

import numpy as np
import cv2
from omegaconf import OmegaConf
import carla
from shapely.geometry import Point, Polygon

# from carla_env import CarlaEnv
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *


def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])


def collect_trajectory(env, save_dir, speed=.5, max_path_length=5000):
    now = datetime.datetime.now()
    salt = np.random.randint(100)
    fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
    save_path = os.path.join(save_dir, fname)
    rgb_path = os.path.join(save_path, 'rgb')
    segmentation_path = os.path.join(save_path, 'segmentation')
    topdown_path = os.path.join(save_path, 'topdown')
    reward_path = os.path.join(save_path, 'reward')
    world_path = os.path.join(save_path, 'world')
    measurements_path = os.path.join(save_path, 'measurements')

    # check for conflicts
    if os.path.isdir(save_path):
        print('Directory conflict, trying again...')
        return 0
    
    # make directories
    os.mkdir(save_path)
    os.mkdir(rgb_path)
    os.mkdir(segmentation_path)
    os.mkdir(topdown_path)
    os.mkdir(measurements_path)

    obs = env.reset()

    for _ in range(25):
        action = env.get_autopilot_action(speed)
        _ = env.step(action)

    for step in range(max_path_length):
        if args.use_autopilot:
            action = env.get_autopilot_action(speed)
        else:
            action = np.random.uniform([-.5,-1.],[.5,1.], (2,))

        if args.use_autopilot and args.apply_noise:
            noise = 1e-2 * np.random.randn()
            action[0] += noise
            action[0] = np.clip(-1,1,action[0])

        next_obs, reward, done, info = env.step(action)

        rgb = info['sensor.camera.rgb/front']
        segmentation = info['sensor.camera.semantic_segmentation/top']
        topdown = info['sensor.camera.rgb/top']

        keys = list(info.keys())
        for key in keys:
            if sys.getsizeof(info[key]) > 2500:
                del info[key]
            elif isinstance(info[key], np.ndarray):
                info[key] = info[key].tolist()

        experience = {
            'obs': obs.tolist(),
            'next_obs': next_obs.tolist(),
            'action': action.tolist(),
            'reward': reward,
            'done': done.item()
        }
        experience.update(info)

        save_env_state(rgb, segmentation, topdown, experience, save_path, step)

        if done:
            break

        obs = next_obs

    return step + 1


def save_env_state(rgb, segmentation, topdown, measurements, save_path, idx):
    rgb_path = os.path.join(save_path, 'rgb', '{:04d}.png'.format(idx))
    cv2.imwrite(rgb_path, rgb)

    segmentation_path = os.path.join(save_path, 'segmentation', '{:04d}.png'.format(idx))
    segmentation = segmentation.argmax(axis=-1)
    cv2.imwrite(segmentation_path, segmentation)

    topdown_path = os.path.join(save_path, 'topdown', '{:04d}.png'.format(idx))
    cv2.imwrite(topdown_path, topdown)

    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def transform_to_list(transform):
    location, rotation = transform.location, transform.rotation
    out = [location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll]
    return out


def main(args):
    config = DefaultMainConfig()

    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/top'] = {
        'x':0.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'64',
        'sensor_y_res':'64',
        'fov':'90', \
        'sensor_tick': '0.0'}

    scenario_config = NoCrashDenseTown01Config() # LeaderboardConfig()
    scenario_config.city_name = args.town
    scenario_config.num_pedestrians = 50
    scenario_config.sample_npc = True
    scenario_config.num_npc_lower_threshold = 50
    scenario_config.num_npc_upper_threshold = 150

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(observation_config=obs_config, scenario_config=scenario_config)
    config.server_fps = 20
    config.carla_gpu = args.gpu

    env = CarlaEnv(config=config, log_dir=args.path + '/')
    try:
        total_samples = 0
        while total_samples < args.n_samples:
            traj_length = collect_trajectory(env, args.path, args.speed)
            total_samples += traj_length
        print('Collected {} samples'.format(total_samples))
    finally:
        env.close()
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--path', type=str)
    parser.add_argument('--use_autopilot', type=int, default=1)
    parser.add_argument('--apply_noise', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    main(args)
