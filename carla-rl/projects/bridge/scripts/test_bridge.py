from projects.bridge.algorithm.bridge_env import LavaBridge

from common.loggers.comet_logger import CometLogger
from projects.bridge.config.logger_config import ExistingCometLoggerConfig

from algorithms import PPO, BASAC


import cv2
import numpy as np
import os
import argparse
import subprocess
import shutil

def generate_video(logger, image_path, save_path, name):
        vid_path = os.path.join(save_path, name + '.mp4')

        im_path = os.path.join(image_path, "%04d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path , "-framerate", '10', "-pix_fmt", "yuv420p",
        vid_path]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        gen_vid_process.wait()


        # logger.log_asset("dynamics_eval/videos", vid_path)

        # Clear the temporary directory of images after video is generated
        shutil.rmtree(image_path)
        os.mkdir(image_path)

def main(args):
    device = f'cuda:{args.gpu}'

    # Construct logger and retrieve policy
    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = args.exp_key

    logger = CometLogger(logger_conf)
    model_path = logger.other_load("policy/models", args.policy_name)

    policy = BASAC.load(model_path, device = device)

    image_save_dir = logger.prep_dir(os.path.join("rollouts/images", logger.experiment_name))
    video_save_dir = logger.prep_dir(os.path.join("rollouts/videos", logger.experiment_name))

    env = LavaBridge()

    obs = env.reset()

    i = 0
    done = False
    ent_coef = np.array([1.00000000e-04])
    print(ent_coef)
    cum_reward = 0
    while not done:
        obs = np.concatenate([obs, ent_coef])
        action, _ = policy.predict(obs, deterministic=False)
        action[0] = 0
        action[1] = 1.0
        # action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        img = env.render()

        cv2.imwrite(os.path.join(image_save_dir, f"{i:04d}.png"), img)

        i += 1
        cum_reward += reward

    print(f"Final Reward {cum_reward}")

    generate_video(logger, image_save_dir, video_save_dir, "rollout")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--exp_key', type=str)
    parser.add_argument('--policy_name', type=str)
    args = parser.parse_args()

    main(args)

