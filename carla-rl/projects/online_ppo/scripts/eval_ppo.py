import sys
import os
import shutil
import argparse
import carla
import subprocess
from tqdm import tqdm
import cv2

import faulthandler
faulthandler.enable()

from common.loggers.comet_logger import CometLogger
from projects.online_ppo.config.logger_config import ExistingCometLoggerConfig

from algorithms import PPO, SAC

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig


class PPOEvaluationConf:
    def __init__(self):
        self.policy_model_name = "best_model_500000.zip"
        self.experiment_key = "34b0ebe1b33a4da38b9f94b46672fc85"

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



def generate_rollouts(logger, env, policy, n_rollouts = 25, timeout = 5000):
    image_save_dir = logger.prep_dir("policy_eval/images")
    video_save_dir = logger.prep_dir("policy_eval/videos")

    global_idx = 0
    success_count = 0
    for rollout in range(n_rollouts):
        print(f"Rollout #{rollout}")

        obs = env.reset(unseen = True, index = rollout)
        for i in tqdm(range(timeout)):
            action, _ = policy.predict(obs)

            obs, reward, done, info = env.step(action)

            image = info["sensor.camera.rgb/top"]

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(image_save_dir, "{:04d}.png".format(global_idx)), image)

            global_idx += 1

            if done:
                success_count += int(info['termination_state'] == "success")
                break

                pass

    print(f"Success rate: {success_count / n_rollouts}")
    generate_video(logger = logger,
                    image_path = image_save_dir,
                    save_path = video_save_dir,
                    name = "rollouts.mp4")


def main(args):
    ppo_evaluation_conf = PPOEvaluationConf()


    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = ppo_evaluation_conf.experiment_key

    device = f"cuda:{args.gpu}"

    logger =  CometLogger(logger_conf)
    print(logger.log_dir)

    print("PPO: Loading policy {}".format(ppo_evaluation_conf.policy_model_name))

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashDenseTown01Config",
        testing = False,
        carla_gpu = args.gpu
    )
    # config.verify()
    # logger_callback = PPOLoggerCallback(logger)

    env = CarlaEnv(config = config, logger = None, log_dir = logger.log_dir)

    policy = PPO.load(
                logger.other_load("policy/models", ppo_evaluation_conf.policy_model_name),
                device = device)


    generate_rollouts(logger = logger, env = env, policy = policy, n_rollouts = 25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    main(args)

