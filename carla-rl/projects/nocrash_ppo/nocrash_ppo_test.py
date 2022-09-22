import sys
import os
import carla
import gym
from algorithms import PPO, SAC
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import time

def main():
    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationNoCameraConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashRegularTown01Config",
        carla_gpu = 0,
        render_server = True
    )

    env = CarlaEnv(config = config)
    policy = PPO.load('checkpoints/model/policy_checkpoint__1000_steps.zip', device = 0)

    #25 routes for Town 1 and Town 2
    episodes = 25
    success = 0
    for i in range(episodes):
        obs = env.reset(unseen = True, index = i)
        done = False
        while(not done):
            action,_ = policy.predict(obs)
            obs, reward, done, info = env.step(action)
        success += int(info['termination_state'] == "success")
        print('SUCCESS RATE: ',success,'/',i+1)

if __name__ == '__main__':
    main()