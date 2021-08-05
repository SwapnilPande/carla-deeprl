import carla
import cv2
import os
from tqdm import tqdm
import argparse
import subprocess
import shutil

import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import VehicleDynamicsOnlyConfig


# Logger
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig


# MOPO
from projects.morel_mopo.algorithm.mopo import MOPO




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


class MOPOEvaluationConf:
    def __init__(self):
        self.model_name = "best_model_1310000.zip"
        self.experiment_key = "08c1ec38a6a643b0a42e8646e1a167e9"
        self.policy_only = True


def generate_rollouts(logger, env, policy, n_rollouts = 10, timeout = 1000):
    image_save_dir = logger.prep_dir("policy_eval/images")
    video_save_dir = logger.prep_dir("policy_eval/videos")

    global_idx = 0

    for rollout in range(n_rollouts):
        print(f"Rollout #{rollout}")

        obs = env.reset()
        for i in tqdm(range(timeout)):
            action = policy.policy_predict(obs)

            obs, reward, done, info = env.step(action)

            image = info["sensor.camera.rgb/top"]

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(image_save_dir, "{:04d}.png".format(global_idx)), image)

            global_idx += 1

    generate_video(logger = logger,
                    image_path = image_save_dir,
                    save_path = video_save_dir,
                    name = "rollouts.mp4")





def main(args):
    # First, set up comet logger to retrieve experiment
    mopo_evaluation_conf = MOPOEvaluationConf()

    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = mopo_evaluation_conf.experiment_key

    logger = CometLogger(logger_conf)

    mopo = MOPO.load(logger = logger,
                    model_name = mopo_evaluation_conf.model_name,
                    gpu = args.gpu,
                    policy_only = mopo_evaluation_conf.policy_only)

    mopo.config.eval_env_config.carla_gpu = args.gpu
    mopo.config.eval_env_config.obs_config = VehicleDynamicsOnlyConfig()
    env = CarlaEnv(config = mopo.config.eval_env_config, logger = logger, log_dir = logger.log_dir)

    generate_rollouts(logger = logger, env = env, policy = mopo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--n_samples', type=int, default=100000)
    # parser.add_argument('--behavior', type=str, default='cautious')
    # parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)





# config = DefaultMainConfig()
# config.populate_config(
#     observation_config = "VehicleDynamicsNoCameraConfig",
#     action_config = "MergedSpeedTanhConfig",
#     reward_config="Simple2RewardConfig",
#     scenario_config = "NoCrashEmptyTown01Config",
#     testing = False,
#     carla_gpu = 1
# )
# # logger_callback = PPOLoggerCallback(logger)


# env = CarlaEnv(config = config, logger = None, log_dir = "/home/swapnil/carla_logs")


# policy = PPO.load("/home/swapnil/best_model.zip")

# while True:
#     obs = env.reset()
#     for i in range(1000):
#         action, _states = policy.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         env.render()
