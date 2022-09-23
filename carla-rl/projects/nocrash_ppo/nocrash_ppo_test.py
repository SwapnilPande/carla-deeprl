import sys
import os
import carla
import numpy as np
import gym
from algorithms import PPO, SAC
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import time
import cv2

def main():
    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashRegularTown01Config",
        carla_gpu = 0,
        render_server = True
    )

    env = CarlaEnv(config = config)
    policy = PPO.load('checkpoints/policy_checkpoint__1100_steps', device = 0)

    #25 routes for Town 1 and Town 2
    episodes = 25
    success = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for i in range(episodes):
        obs = env.reset(unseen = True, index = i)
        done = False
        video=cv2.VideoWriter('videos/'+str(i)+'.mp4',fourcc,10,(512,512))
        while(not done):
            action,_ = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            img = info["sensor.camera.rgb/top"]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.write(img)            
        video.release()
        success += int(info['termination_state'] == "success")
        print('SUCCESS RATE: ',success,'/',i+1)

if __name__ == '__main__':
    main()