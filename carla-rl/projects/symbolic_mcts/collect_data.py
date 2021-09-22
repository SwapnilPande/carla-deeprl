import sys
import os
import argparse
import json
import datetime
import math

import numpy as np
import carla
import cv2

from environment import CarlaEnv
from projects.symbolic_mcts.symbolic_env import SymbolicCarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.algorithm.mopo import MOPO
from projects.morel_mopo.config.morel_mopo_config import DefaultMLPMOPOConfig, DefaultProbMLPMOPOConfig, DefaultProbGRUMOPOConfig, DefaultMLPObstaclesMOPOConfig


def collect_trajectory(env, save_dir, speed=5.0, max_path_length=2500):
    now = datetime.datetime.now()
    salt = np.random.randint(100)
    fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
    save_path = os.path.join(save_dir, fname)

    # check for conflicts
    if os.path.isdir(save_path):
        print('Directory conflict, trying again...')
        return 0
    
    # make directories
    os.mkdir(save_path)

    frames = []

    obs = env.reset()

    for _ in range(10):
        action = env.get_autopilot_action(speed)
        _ = env.step(action)

    for step in range(max_path_length):
        if args.use_autopilot:
            action = env.get_autopilot_action(speed)
        else:
            action = np.random.uniform([-.5,-1.],[.5,1.], (2,))

        if args.use_autopilot and args.apply_noise:
            noise = 1e-2 * np.random.randn()
            _action = action.copy()
            _action[0] += noise
            _action[0] = np.clip(-1,1,_action[0])
        else:
            _action = action

        next_obs, reward, done, info = env.step(_action)

        frame = env.render()
        frames.append(frame)

        experience = {
            'obs': obs,
            'next_obs': next_obs,
            'action': action.tolist(),
            'reward': reward,
            'done': done.item(),
        }
        save_env_state(experience, save_path, step)

        if done:
            break

        obs = next_obs

    print(info['termination_state'])

    print('Saving to {}'.format(save_path))
    save_video(frames, save_path + '/rollout.avi')

    return step + 1


def save_video(frames, fname, fps=15):
    frames = [np.array(frame) for frame in frames]
    height, width = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()


def save_env_state(measurements, save_path, idx):
    measurements_path = os.path.join(save_path, '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def transform_to_list(transform):
    location, rotation = transform.location, transform.rotation
    out = [location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll]
    return out


def main(args):
    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "VehicleDynamicsObstacleNoCameraConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashDenseTown02Config",
        testing = False,
        carla_gpu = args.gpu
    )

    config.obs_config.sensors['sensor.camera.rgb/front'] = {
        'x':2.0,
        'z':1.4,
        'pitch':0.0,
        'sensor_x_res':'128',
        'sensor_y_res':'128',
        'fov':'120',
        'sensor_tick': '0.0'
    }
    config.render_server = True

    env = SymbolicCarlaEnv(config=config)
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
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--path', type=str)
    parser.add_argument('--use_autopilot', type=int, default=1)
    parser.add_argument('--apply_noise', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    main(args)
