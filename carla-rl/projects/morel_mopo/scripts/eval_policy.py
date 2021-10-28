import carla
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import os
from tqdm import tqdm
import argparse
import subprocess
import shutil
import copy

# Logger
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig

import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import VehicleDynamicsOnlyConfig, VehicleDynamicsObstacleConfig
from environment.config.scenario_configs import NoCrashRegularTown01Config, NoCrashDenseTown01Config





# MOPO
from projects.morel_mopo.algorithm.mopo import MOPO

class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()

    def policy_predict(self, obs):
        return self.env.get_autopilot_action()





def generate_video(logger, image_path, save_path, name):
        vid_path = os.path.join(save_path, name + '.mp4')

        im_path = os.path.join(image_path, "%04d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path , "-framerate", '20', "-pix_fmt", "yuv420p",
        vid_path]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        gen_vid_process.wait()


        logger.log_asset("policy_eval/videos", vid_path)

        # Clear the temporary directory of images after video is generated
        shutil.rmtree(image_path)
        os.mkdir(image_path)



def generate_rollouts(logger, env, policy, n_rollouts = 10, timeout = 500):
    image_save_dir = logger.prep_dir("policy_eval/images")
    video_save_dir = logger.prep_dir("policy_eval/videos")

    global_idx = 0

    for rollout in range(n_rollouts):
        print(f"Rollout #{rollout}")

        obs = env.reset()
        for i in tqdm(range(timeout)):
            action = policy.policy_predict(obs)

            obs, reward, done, info = env.step(action)
            print(obs[0][3])

            image = info["sensor.camera.rgb/top"]

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(image_save_dir, "{:04d}.png".format(global_idx)), image)

            global_idx += 1

            if done:
                break
                pass

    generate_video(logger = logger,
                    image_path = image_save_dir,
                    save_path = video_save_dir,
                    name = "rollouts.mp4")


def closed_loop_eval(exp_name, logger, env, fake_env, policy, n_rollouts):
    """Generates images plotting multiple trajectories sampled from the dynamics model against ground truth
    """
    # Get the number of stacked frames fake_env needs
    frame_stack = fake_env.frame_stack

    image_save_dir = logger.prep_dir(os.path.join("closed_loop_eval/images", exp_name))

    total_steps = 0
    real_done = False
    for i in range(n_rollouts):

        real_rollout_steps = 0
        img_idx = 0

        print("Experiment: {}, Episode {}".format(exp_name, i))
        real_dynamics_obs_history = deque([], maxlen = frame_stack)
        real_dynamics_action_history = deque([], maxlen = frame_stack)
        real_dynamics_pose = None

        # Reset environment and get real observation
        real_obs = env.reset()
        # Do this step because we need to data returned in info, which reset does not return
        real_obs, real_reward, real_done, real_info = env.step(np.array([0,0]))


        # Since the fake environment requires a frame stack input, we have to stack up the frames
        # Loop over frame stack to collect the correct rollout stack to start using the fake_env
        for frame in range(frame_stack):
            real_action = policy.policy_predict(real_obs)
            real_dynamics_action_history.appendleft(real_action)

            real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))

            real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                            real_info["ego_vehicle_y"],
                            real_info["ego_vehicle_theta"]])

            real_obs, real_reward, real_done, real_info = env.step(real_action)

        real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))

        real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                            real_info["ego_vehicle_y"],
                            real_info["ego_vehicle_theta"]])

        real_done = False
        fake_done = False
        real_rollout_steps = 0
        fake_rollout_steps = 0

        img_idx = 0

        while not real_done and not fake_done:
            real_rollout_pose = [real_dynamics_pose]
            real_rollout_obs = [real_dynamics_obs_history[0]]
            real_policy_obs = [real_obs]

            fake_rollout_pose = [real_dynamics_pose]
            fake_rollout_obs = [real_dynamics_obs_history[0]]
            fake_policy_obs = [real_obs]

            init_real_dynamics_obs_history = copy.deepcopy(real_dynamics_obs_history)
            init_real_action_history = copy.deepcopy(real_dynamics_action_history)
            init_dynamics_pose = copy.deepcopy(real_dynamics_pose)
            init_waypoints = real_info["waypoints"]


            for rollout_step in range(fake_env.timeout_steps-1):
                action = policy.policy_predict(real_obs)

                real_obs, real_reward, real_done, real_info = env.step(action)

                real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                                                    real_info["ego_vehicle_y"],
                                                    real_info["ego_vehicle_theta"]])

                if(rollout_step == 0):
                    start_image = real_info["sensor.camera.rgb/top"]
                if(rollout_step == fake_env.timeout_steps-2):
                    end_image = real_info["sensor.camera.rgb/top"]

                real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))
                real_dynamics_action_history.appendleft(action)

                real_rollout_pose.append(real_dynamics_pose)
                real_policy_obs.append(real_obs)

                if(real_done):
                    end_image = real_info["sensor.camera.rgb/top"]
                    break

            print("Writing image")
            combined = cv2.hconcat([start_image, end_image])
            combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(image_save_dir, "sim_{:04d}.png".format(img_idx)), combined)

            fake_obs = fake_env.reset(inp = (
                        np.array(init_real_dynamics_obs_history),
                        np.array(init_real_action_history),
                        init_dynamics_pose,
                        init_waypoints
                    ))
            import ipdb; ipdb.set_trace()
            for rollout_step in range(fake_env.timeout_steps-1):
                action = policy.policy_predict(np.expand_dims(fake_obs, axis = 0))

                fake_obs, fake_reward, fake_done, fake_info = fake_env.step(action)

                fake_rollout_pose.append(fake_env.vehicle_pose.cpu().numpy())
                fake_policy_obs.append(fake_obs)

                if(fake_done):
                    break

            real_rollout_pose = np.array(real_rollout_pose)
            fake_rollout_pose = np.array(fake_rollout_pose)

            real_x = real_rollout_pose[:,0]
            real_y = real_rollout_pose[:,1]
            fake_x = fake_rollout_pose[:,0]
            fake_y = fake_rollout_pose[:,1]

            plt.figure()
            plt.plot(real_x, real_y, color = "blue", label = "Real Env")
            plt.plot(fake_x, fake_y, color = "orange", label = "Dyn Env")

            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()

            delta_x = x_max - x_min
            if(delta_x < 4):
                extra = (4 - delta_x)/2
                x_min -= extra
                x_max += extra

            delta_y = y_max - y_min
            if(delta_y < 4):
                extra = (4 - delta_y)/2
                y_min -= extra
                y_max += extra

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            plt.legend()

            plt.savefig(os.path.join(image_save_dir, "{:04d}.png".format(img_idx)))
            plt.close()

            img_idx += 1

            print(real_done)
            print(fake_done)


class MOPOEvaluationConf:
    def __init__(self):
        self.policy_model_name = "best_model_1800000.zip"
        self.experiment_key = "aa2787f0101e4a119906ca92fc0c5838"
        self.policy_only = True
        # self.dynamics_model_name = "final"
        self.dynamics_model_name = None


def main(args):
    # First, set up comet logger to retrieve experiment
    mopo_evaluation_conf = MOPOEvaluationConf()

    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = mopo_evaluation_conf.experiment_key

    logger = CometLogger(logger_conf)

    mopo = MOPO.load(logger = logger,
                    policy_model_name = mopo_evaluation_conf.policy_model_name,
                    gpu = args.gpu,
                    policy_only = mopo_evaluation_conf.policy_only,
                    dynamics_model_name = mopo_evaluation_conf.dynamics_model_name)


    mopo.config.eval_env_config.render_server = True
    mopo.config.eval_env_config.carla_gpu = args.gpu
    mopo.config.eval_env_config.obs_config = VehicleDynamicsObstacleConfig() # VehicleDynamicsOnlyConfig()
    mopo.config.eval_env_config.scenario_config = NoCrashDenseTown01Config()
    env = CarlaEnv(config = mopo.config.eval_env_config, logger = logger, log_dir = logger.log_dir)

    # autopilot_policy = AutopilotPolicy(env)
    # closed_loop_eval("test", logger, env, mopo.fake_env, mopo, 5)

    generate_rollouts(logger = logger, env = env, policy = mopo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--n_samples', type=int, default=100000)
    # parser.add_argument('--behavior', type=str, default='cautious')
    # parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
