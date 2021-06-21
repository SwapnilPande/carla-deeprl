import os
import sys
import json
import datetime
import argparse
from tqdm import tqdm

import numpy as np
import cv2

from gym.core import ObservationWrapper
# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback



# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
EXPERIMENT_NAME = "NO_CRASH_EMPTY_FIRST_TEST"

logger_conf = CometLoggerConfig()
logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["Online_PPO"])


class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()

    
class AutopilotRandomPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.action_space.sample()


class AutopilotNoisePolicy:
    def __init__(self, env, steer_noise_std, speed_noise_std):
        self.env = env
        self.steer_noise_std = steer_noise_std
        self.speed_noise_std = speed_noise_std

    def __call__(self, obs):
        res = self.env.get_autopilot_action()
        res[0] += np.random.normal(loc=0.0, scale=self.steer_noise_std, size=1)[0]
        res[1] += np.random.normal(loc=0.0, scale=self.speed_noise_std, size=1)[0]
        return res

    
def collect_trajectory(env, save_dir, policy, max_path_length=5000):
    now = datetime.datetime.now()
    salt = np.random.randint(100)

    fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
    save_path = os.path.join(save_dir, fname)
    # rgb_path = os.path.join(save_path, 'rgb')
    # topdown_path = os.path.join(save_path, 'topdown')
    measurements_path = os.path.join(save_path, 'measurements')

    if os.path.isdir(save_path):
        print('Directory conflict, trying again...')
        return 0

    # make directories
    os.mkdir(save_path)
    # os.mkdir(rgb_path)
    # os.mkdir(topdown_path)
    os.mkdir(measurements_path)

    seed = np.random.randint(10000000)
    obs = env.reset()

    for step in tqdm(range(max_path_length)):


        action = policy(obs)

        next_obs, reward, done, info = env.step(action)
        experience = {
            'obs': obs.tolist(),
            'next_obs': next_obs.tolist(),
            'action': action.tolist(),
            'reward': reward,
            'done': done
        }
        experience.update(info)

        save_env_state(experience, save_path, step)

        if done:
            break

        obs = next_obs

    return step + 1


def save_env_state(measurements, save_path, idx, rgb = None, topdown = None):
    if rgb is not None:
        rgb_path = os.path.join(save_path, 'rgb', '{:04d}.png'.format(idx))
        cv2.imwrite(rgb_path, rgb)

    if topdown is not None:
        topdown_path = os.path.join(save_path, 'topdown', '{:04d}.png'.format(idx))
        cv2.imwrite(topdown_path, topdown)

    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def main(args):

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationNoCameraConfig",
        action_config = "MergedSpeedTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashEmptyTown01Config",
        testing = False,
        carla_gpu = args.gpu
    )


    with CarlaEnv(config = config, log_dir = "/home/scratch/swapnilp/carla_test") as env:
        # Create the policy
        policy = AutopilotPolicy(env)

        total_samples = 0
        while total_samples < args.n_samples:
            traj_length = collect_trajectory(env, args.path, policy)
            total_samples += traj_length

    print('Done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_samples', type=int, default=100000)
    #parser.add_argument('--behavior', type=str, default='cautious')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
