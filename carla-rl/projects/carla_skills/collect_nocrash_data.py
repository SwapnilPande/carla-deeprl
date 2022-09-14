import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join('../../../')))
import numpy as np
import gym
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        res = self.env.get_autopilot_action()
        res[0] += np.random.normal(loc=0.0, scale=0.06, size=1)[0]
        res[1] += np.random.normal(loc=0.0, scale=0.06, size=1)[0]
        return res
        #return self.env.get_autopilot_action()


if __name__ == "__main__":
    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LeaderboardObsConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashDenseTown01Config",
        testing = False,
        carla_gpu = 0,
        render_server=False
    )

    env = CarlaEnv(config = config)
    policy = AutopilotPolicy(env)
    data = {}
    #data['observations'] = np.load('data/noisy_observations.npy')
    #data['next_observations'] = np.load('data/noisy_next_observations.npy')
    #data['actions'] = np.load('data/noisy_actions.npy')
    #data['terminals'] = np.load('data/noisy_terminals.npy')
    episodes = 600

    for eps in range(episodes):
        print('EPISODE ',eps,':')
        obs = env.reset()
        if(eps==0):
            data['observations'] = np.zeros((0,obs.shape[1]+3))
            data['actions'] = np.zeros((0,2))
            data['next_observations'] = np.zeros((0,obs.shape[1]+3))
            data['terminals'] = np.zeros((0,1))
        done = False
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        pos_x = env.episode_measurements['ego_vehicle_x']
        pos_y = env.episode_measurements['ego_vehicle_y']
        pos_theta = env.episode_measurements['ego_vehicle_theta']

        while(not done):
            observations = np.concatenate((obs[0],np.array([pos_x]),np.array([pos_y]),np.array([pos_theta])))
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            pos_x = env.episode_measurements['ego_vehicle_x']
            pos_y = env.episode_measurements['ego_vehicle_y']
            pos_theta = env.episode_measurements['ego_vehicle_theta']
            next_observations = np.concatenate((obs[0],np.array([pos_x]),np.array([pos_y]),np.array([pos_theta])))
            data['observations'] = np.vstack([data['observations'],observations])
            data['actions'] = np.vstack([data['actions'],action])
            data['next_observations'] = np.vstack([data['next_observations'],next_observations])
            data['terminals'] = np.vstack([data['terminals'],np.array([done[0]])])

        print('SAVING DATA FROM EPISODE')
        np.save('/home/hyperpotato/carla-deeprl/carla-rl/projects/carla_skills/data/observations2.npy',data['observations'])
        np.save('/home/hyperpotato/carla-deeprl/carla-rl/projects/carla_skills/data/actions2.npy',data['actions'])
        np.save('/home/hyperpotato/carla-deeprl/carla-rl/projects/carla_skills/data/observations2.npy',data['next_observations'])
        np.save('/home/hyperpotato/carla-deeprl/carla-rl/projects/carla_skills/data/terminals2.npy',data['terminals'])
        print('DATASET LENGTH: ',data['observations'].shape[0])