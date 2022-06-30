import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join('../../')))
import gym
# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()

if __name__ == "__main__":

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "VehicleDynamicsNoCameraConfig",
        action_config = "MergedSpeedScaledTanhSpeed40Config",
        reward_config = "Simple2RewardConfig",
        scenario_config = "SimpleSingleTurnConfig",
        testing = False,
        carla_gpu = 0
    )

    env = CarlaEnv(config = config)
    policy = AutopilotPolicy(env)

    episodes = 10

    obs = env.reset()
    total_reward = 0

    for eps in range(episodes):
        ep_reward = 0
        while(True):
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)
            ep_reward = ep_reward + reward
            experience = {
                        'obs': obs.tolist(),
                        'next_obs': next_obs.tolist(),
                        'action': action.tolist(),
                        'reward': reward,
                        'done': done.tolist()
                    }
            experience.update(info)
            obs = next_obs

            if(done):
                print('Episode ',str(eps),' reward = ',ep_reward)
                total_reward += ep_reward
                obs = env.reset()
                break
    mean_reward = total_reward/episodes
    print('Mean reward = ',mean_reward)