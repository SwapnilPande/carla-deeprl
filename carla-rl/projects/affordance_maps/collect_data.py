import os
import json
import datetime
import argparse
import random

import numpy as np
import cv2
from omegaconf import OmegaConf

# from carla_env import CarlaEnv
from environment import CarlaEnv
# from agents.navigation.behavior_agent import BehaviorAgent



def collect_trajectory(env, save_dir, speed=.5, max_path_length=5000):
    now = datetime.datetime.now()
    salt = np.random.randint(100)
    fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
    save_path = os.path.join(save_dir, fname)
    rgb_path = os.path.join(save_path, 'rgb')
    segmentation_path = os.path.join(save_path, 'segmentation')
    topdown_path = os.path.join(save_path, 'topdown')
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

    # calibration = np.array([[64, 0, 64],
    #                         [0, 64, 64],
    #                         [0,  0,  1]])
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/top'].sensor
    # camera_actor.calibration = calibration

    for step in range(max_path_length):
        action = env.get_autopilot_action(speed)
        next_obs, reward, done, info = env.step(action)

        rgb = info['rgb_front']
        segmentation = info['sem_bev']
        topdown = info['rgb_bev']

        experience = {
            'obs': obs.tolist(),
            'next_obs': next_obs.tolist(),
            'action': action.tolist(),
            'reward': reward,
            'done': done.item(),

            'actor_tf': transform_to_list(ego_actor.get_transform()),
            'camera_tf': transform_to_list(camera_actor.get_transform()),

            'speed': info['speed']
        }

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
    env = CarlaEnv(log_dir='{}'.format(args.path))
    try:
        total_samples = 0
        while total_samples < args.n_samples:
            traj_length = collect_trajectory(env, args.path, args.speed)
            total_samples += traj_length
    finally:
        env.close()
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
