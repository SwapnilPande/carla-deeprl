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
        testing = False,
        carla_gpu = 0
    )

    env = CarlaEnv(config = config)
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path='checkpoints/model',
                                                name_prefix='policy_checkpoint_')

    model = PPO("MlpPolicy", env, verbose=1, device = 0)


    callbacks = [checkpoint_callback]
    model.learn(total_timesteps=10000000, callback = callbacks)

    model.save('checkpoints/final_model')

if __name__ == '__main__':
    main()