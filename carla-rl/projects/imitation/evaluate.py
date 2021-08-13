import os

import numpy as np
import matplotlib.pyplot as plt
import hydra
import torch
import cv2

from models import ConvAgent, RecurrentAttentionAgent
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *


EXPERIMENT_DIR = '/home/scratch/brianyan/outputs/resnet+lstm/2021-08-10_14-42-04'
CHECKPOINT = 'epoch=75-step=118787.ckpt' # 'epoch=34-step=1097179.ckpt' # 'epoch=66-step=104720.ckpt'


def save_video(frames, fname, fps=15):
    frames = [np.array(frame) for frame in frames]
    height, width = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()


@hydra.main(config_name='{}/.hydra/config.yaml'.format(EXPERIMENT_DIR))
def main(cfg):
    # agent = hydra.utils.instantiate(cfg.algo.agent)
    agent = RecurrentAttentionAgent.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT), **cfg.agent) # ConvAgent.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT))
    agent = agent.cuda().eval()

    reward_list = []
    status_list = []

    config = DefaultMainConfig()

    obs_config = LowDimObservationConfig()
    # obs_config.sensors['sensor.camera.rgb/top'] = {
    #     'x':0.0,
    #     'z':18.0,
    #     'pitch':270,
    #     'sensor_x_res':'64',
    #     'sensor_y_res':'64',
    #     'fov':'90', \
    #     'sensor_tick': '0.0'}

    scenario_config = NoCrashDenseTown01Config() # LeaderboardConfig()
    scenario_config.city_name = 'Town02'
    # scenario_config.num_pedestrians = 50
    scenario_config.sample_npc = True
    scenario_config.num_npc_lower_threshold = 65
    scenario_config.num_npc_upper_threshold = 75

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(observation_config=obs_config, scenario_config=scenario_config)
    config.server_fps = 20
    config.carla_gpu = cfg.gpu

    env = CarlaEnv(config=config)

    frames = []

    try:
        for index in range(25):
            obs = env.reset(unseen=True, index=index)
            agent.reset()
            total_reward = 0.

            for i in range(5000):
                image_obs = env.render(camera='sensor.camera.rgb/front')
                with torch.no_grad():
                    action = agent.predict(image_obs, obs)
                obs, reward, done, info = env.step(action[0])

                frames.append(image_obs)

                # print(action, reward)
                total_reward += reward

                # # import ipdb; ipdb.set_trace()
                # if i % 20 == 0:
                #     fig, axs = plt.subplots(2)
                #     attention_map = attention_map[0].mean(axis=-1)
                #     attention_map = attention_map / attention_map.max()
                #     axs[0].imshow(attention_map)
                #     axs[1].imshow(image_obs)
                #     plt.savefig('/zfsauton2/home/brianyan/carla-rl/carla-rl/projects/imitation/maps/{}.png'.format(i))
                #     plt.close()

                if done:
                    break

            reward_list.append(total_reward)
            status_list.append(info['termination_state'])
            print(status_list[-1])

            # cv2.destroyAllWindows()
    finally:
        env.close()

    video_path = os.path.join(os.getcwd(), 'evaluation.avi')
    save_video(frames, video_path)

if __name__ == '__main__':
    main()
