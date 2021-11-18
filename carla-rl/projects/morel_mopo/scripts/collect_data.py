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
# from projects.morel_mopo.config.logger_config import CometLoggerConfig
from algorithms.restart_wrapper import collect_data_restart_wrapper

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import VecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
EXPERIMENT_NAME = "NO_CRASH_EMPTY_FIRST_TEST"

# logger_conf = CometLoggerConfig()
# logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["Online_PPO"])



class NumpyEncoder(json.JSONEncoder):
    """ json encoder for numpy array """
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()


class RandomPolicy:
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


class DataCollector():
    def __init__(self):
        self.env = None
        self.policy = None
        self.path = None

    def collect_data(self, env = None,
                           path=None,
                           policy=None,
                           n_samples=1,
                           carla_gpu=0,
                    ):

        assert(path is not None), "path cannot be None"
        self.path = path

        print('************using path', self.path)

        # Set env if not passed in
        self.env = env
        if self.env is None:
            config = DefaultMainConfig()
            config.populate_config(
                observation_config = "VehicleDynamicsNoCameraConfig",
                action_config = "MergedScaledSpeedTanhConfig",
                reward_config = "Simple2RewardConfig",
                scenario_config = "NoCrashEmptyTown01Config",
                testing = False,
                carla_gpu = carla_gpu
            )
            self.env = CarlaEnv(config = config, log_dir = "/home/scratch/vccheng/carla_test")

        # Create the policy
        if policy is None:
            self.policy = AutopilotPolicy(self.env)
        else:
            # if policy passed in
            self.policy = policy

        # Collect trajectories until <n_samples>
        total_samples = 0
        while total_samples < n_samples:
            traj_length = self.collect_trajectory()
            total_samples += traj_length

        print('Done collecting data')
        return self


    def collect_trajectory(self, max_path_length=5000, save_to_json=True):

        assert(self.path is not None), "path cannot be None"
        assert(self.policy is not None), "policy cannot be None"
        assert(self.env is not None), "envcannot be None"

        now = datetime.datetime.now()
        salt = np.random.randint(100)

        fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
        save_path = os.path.join(self.path, fname)
        # rgb_path = os.path.join(save_path, 'rgb')
        # topdown_path = os.path.join(save_path, 'topdown')
        measurements_path = os.path.join(save_path, 'measurements')
        print("Saving measurements to: ", measurements_path)

        if save_to_json:
            if os.path.isdir(save_path):
                print('Directory conflict, trying again...')
                return 0
            # make directories
            os.makedirs(save_path)
            # os.mkdir(rgb_path)
            # os.mkdir(topdown_path)
            os.makedirs(measurements_path)

        seed = np.random.randint(10000000)
        print('Resetting env in collect_data')
        obs = self.env.reset()

        for step in tqdm(range(max_path_length)):

            try:
                action, _ = self.policy.predict(obs)
                next_obs, reward, done, info = self.env.step(action)

            except:
                action = self.policy(obs)
                next_obs, reward, done, info = self.env.step(action)

            # print(f'next_obs: {next_obs}, rew: {reward}, done: {done}, info: {info}')
            experience = {
                'obs': obs.tolist(),
                'next_obs': next_obs.tolist(),
                'action': action.tolist(),
                'reward': reward,
                'done': done.tolist()
            }

            experience.update(info)
            # save as json
            if save_to_json:
                self.save_env_state(experience, save_path, step) # TODO add back in


            if done:
                break
            obs = next_obs

        return step + 1


    def save_env_state(self, measurements, save_path, idx, rgb = None, topdown = None):
        if rgb is not None:
            rgb_path = os.path.join(save_path, 'rgb', '{:04d}.png'.format(idx))
            cv2.imwrite(rgb_path, rgb)

        if topdown is not None:
            topdown_path = os.path.join(save_path, 'topdown', '{:04d}.png'.format(idx))
            cv2.imwrite(topdown_path, topdown)

        measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
        with open(measurements_path, 'w') as out:
            json.dump(measurements, out, cls=NumpyEncoder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--policy')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    def collect_data_fn(restart_interval):
        data_collector = DataCollector()

        config = DefaultMainConfig()
        config.populate_config(
                    observation_config = "VehicleDynamicsNoCameraConfig",
                    action_config = "MergedSpeedScaledTanhConfig",
                    reward_config = "Simple2RewardConfig",
                    scenario_config = "LeaderboardConfig",
                    testing = False,
                    carla_gpu = args.gpu
                )
        # TEMPORARY
        # config.action_config.target_speed = 40

        env = CarlaEnv(config = config)

        policy = AutopilotNoisePolicy(env, 0.1, 0.1)
        
        data_collector.collect_data(env = env, policy = policy, path = args.path, n_samples = restart_interval, carla_gpu = args.gpu)

    collect_data_restart_wrapper(collect_data_fn, args.n_samples, 10)
