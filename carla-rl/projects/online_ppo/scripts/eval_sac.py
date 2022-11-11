import sys
import os
import shutil
import argparse
import carla
import subprocess
from tqdm import tqdm
import cv2
import ipdb
import numpy as np
from matplotlib import pyplot as plt

import faulthandler
faulthandler.enable()

from common.loggers.comet_logger import CometLogger
from projects.online_ppo.config.logger_config import ExistingCometLoggerConfig

from algorithms import PPO, SAC

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, **kwargs):
        return (self.env.get_autopilot_action(),None)


class SACEvaluationConf:
    def __init__(self):
        self.policy_model_name = "best_model_400000.zip"
        self.experiment_key = "95b8c825bd16403dabae8bd87a3ba88f"

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
            action, _ = policy.predict(obs,deterministic=True)

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


def generate_rewards(env, policy, n_rollouts = 25, timeout = 5000):
    global_idx = 0
    success_count = 0
    rewards_history,action_history = [],[]
    for rollout in range(n_rollouts):
        print(f"Rollout #{rollout}")
        ep_reward,ep_action= [],[]
        obs = env.reset(unseen = True, index = rollout)
        for i in tqdm(range(timeout)):
            action, _ = policy.predict(obs,deterministic=True)
            ep_action.append(action)

            obs, reward, done, info = env.step(action)
            ep_reward.append(reward)
            global_idx += 1

            if done:
                success_count += int(info['termination_state'] == "success")
                break

                pass
        rewards_history.append(ep_reward)
        action_history.append(ep_action)

    print(f"Success rate: {success_count / n_rollouts}")
    
    return rewards_history,action_history


def get_discounted_returns(reward_array,gamma=0.99):
    discounted_rewards = np.zeros(len(reward_array))
    discounted_rewards[-1] = reward_array[-1]
    for i in range(1,len(reward_array)):
        index = len(reward_array) - 1 - i
        discounted_rewards[index] = reward_array[index] + 0.99*discounted_rewards[index+1]
    return discounted_rewards

def plot_histogram(agent_actions,agent_rewards,autopilot_rewards,prefix=''):
    path = os.path.join('/zfsauton2/home/ishaans','output/images')
    titles = ['steering_sac','target_speed_sac','agent_rewards_sac','autopilot_rewards']
    for i in range(len(titles)):
        if i < 2:
            ep_data = [a[i] for a in agent_actions[0]]
        elif i == 2:
            ep_data = get_discounted_returns(agent_rewards[0])
        else:
            ep_data = get_discounted_returns(autopilot_rewards[0])
        counts, bins = np.histogram(ep_data)
        plt.stairs(counts,bins)
        plt.title(titles[i])
        plt.savefig(os.path.join(path,prefix + titles[i] + '.png'))
        plt.clf()

def main(args):
    sac_evaluation_conf = SACEvaluationConf()


    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = sac_evaluation_conf.experiment_key
    #logger_conf.log_dir = '/zfsauton2/home/ishaans/output'

    device = f"cuda:{args.gpu}"

    logger =  CometLogger(logger_conf)
    print(logger.log_dir)

    print("SAC: Loading policy {}".format(sac_evaluation_conf.policy_model_name))

    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "LowDimObservationConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "SimpleSingleTurnConfig",
        testing = False,
        render_server=True,
        carla_gpu = args.gpu
    )
    # config.verify()
    # logger_callback = PPOLoggerCallback(logger)

    env = CarlaEnv(config = config, logger = None, log_dir = logger.log_dir)

    policy = SAC.load(
                logger.other_load("policy/models", sac_evaluation_conf.policy_model_name),
                device = device)



   
  
    autopilot_policy = AutopilotPolicy(env=env)
    generate_rollouts(logger = logger, env = env, policy = policy, n_rollouts = 2)
    autopilot_rewards,autopilot_actions = generate_rewards(env = env, policy = autopilot_policy, n_rollouts = 2)
    agent_rewards,agent_actions = generate_rewards(env = env, policy = policy, n_rollouts = 1)
    #ipdb.set_trace()
    plot_histogram(agent_actions,agent_rewards,autopilot_rewards,prefix='single_turn_')
    

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    main(args)

