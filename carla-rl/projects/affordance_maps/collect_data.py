import os
import json
import datetime
import argparse
import random

import numpy as np
import cv2
from omegaconf import OmegaConf

# from carla_env import CarlaEnv
from environment.carla_9_4.env import CarlaEnv
from agents.torch.sac import SAC
from agents.torch.utils import COLOR
# from agents.navigation.behavior_agent import BehaviorAgent


EXPERIMENT_DIR = '/home/scratch/brianyan/outputs/2021-04-01_14-36-57'
CHECKPOINT = 'epoch=5-step=5999.ckpt'


def collect_trajectory(env, save_dir, behavior='cautious', max_path_length=5000):
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
    # agent = env._agent

    cfg = OmegaConf.load('{}/.hydra/config.yaml'.format(EXPERIMENT_DIR))
    agent = SAC.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT), **cfg.algo.agent)
    agent = agent.cuda().eval()

    for step in range(max_path_length):
        action = agent.predict(obs)[0] if random.random() > .01 else np.random.uniform(-1, 1, (2,))
        next_obs, reward, done, info = env.step(action)

        rgb = info['render_image']
        segmentation = info['segmentation_image']
        topdown = info['topdown_image']
        # action = np.array(info['action'])

        experience = {
            'obs': obs.tolist(),
            'next_obs': next_obs.tolist(),
            'action': action.tolist(),
            'reward': reward.item(),
            'done': done.item()
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
    cv2.imwrite(segmentation_path, segmentation)

    topdown_path = os.path.join(save_path, 'topdown', '{:04d}.png'.format(idx))
    cv2.imwrite(topdown_path, topdown)

    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def main(args):
    env = CarlaEnv(log_dir='{}'.format(args.path), server_port=args.port)
    total_samples = 0
    while total_samples < args.n_samples:
        traj_length = collect_trajectory(env, args.path, args.behavior)
        total_samples += traj_length

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--behavior', type=str, default='cautious')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
