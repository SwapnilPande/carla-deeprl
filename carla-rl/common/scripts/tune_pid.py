## Script for tuning the PID parameters for the carla environment. This PID controller translates the target speed to throttle and brake.
import sys
import os
import argparse
import carla

import gym
import numpy as np
import matplotlib.pyplot as plt\

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

def main(args):
    device = f"cuda:{args.gpu}"

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "VehicleDynamicsNoCameraConfig",
        action_config = "MergedSpeedScaledTanhSpeed40Config",
        reward_config = "Simple2RewardConfig",
        scenario_config = "LeaderboardConfig",
        testing = False,
        carla_gpu = args.gpu,
        render_server = False
    )

    env = CarlaEnv(config = config)

    while True:
        obs = env.reset()
        done = False

        print('Starting PID tuning...')

        # Get input with Kp, Ki, Kd
        print("Enter Kp, Ki, Kd:")
        Kp, Ki, Kd = input().split()
        Kp = float(Kp)
        Ki = float(Ki)
        Kd = float(Kd)

        # env.carla_interface.actor_fleet.update_pid_parameters(Kp, Ki, Kd)

        timestep = 0
        STEP_DURATION = 250
        speeds = []
        while not done and timestep < STEP_DURATION:
            # Generate step input in speed
            action = np.array([0, 1.0])
            obs, reward, done, info = env.step(action)

            speeds.append(info['speed'])

            timestep += 1

        # Plot speed against increments of 0.1 seconds
        plt.plot(np.arange(0, len(speeds))/10, speeds)
        # Populate PID parameters in title
        plt.title(f"Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
        # Savefig to file
        # Concat args.out with filename
        outfile = os.path.join(args.out, f"{Kp}_{Ki}_{Kd}.png")
        plt.savefig(outfile)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    main(args)

