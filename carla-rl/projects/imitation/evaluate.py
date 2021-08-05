import os

import numpy as np
import matplotlib.pyplot as plt
import hydra

from models import AttentionModel, ConvAgent
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *


EXPERIMENT_DIR = '/home/scratch/brianyan/outputs/conv_bc/2021-07-29_14-54-02/' # '/home/scratch/brianyan/outputs/conv_bc/2021-08-01_22-37-18/'
CHECKPOINT = 'epoch=12-step=183649.ckpt' # 'epoch=48-step=692221.ckpt'



@hydra.main(config_name='{}/.hydra/config.yaml'.format(EXPERIMENT_DIR))
def main(cfg):
    # agent = hydra.utils.instantiate(cfg.algo.agent)
    agent = ConvAgent.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT))
    agent = agent.cuda().eval()

    reward_list = []
    status_list = []

    config = DefaultMainConfig()

    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/top'] = {
        'x':0.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'64',
        'sensor_y_res':'64',
        'fov':'90', \
        'sensor_tick': '0.0'}

    scenario_config = NoCrashDenseTown01Config() # LeaderboardConfig()
    scenario_config.city_name = 'Town02'
    scenario_config.num_pedestrians = 50
    scenario_config.sample_npc = True
    scenario_config.num_npc_lower_threshold = 50
    scenario_config.num_npc_upper_threshold = 150

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(observation_config=obs_config, scenario_config=scenario_config)
    config.server_fps = 20
    config.carla_gpu = cfg.gpu

    env = CarlaEnv(config=config)

    try:
        for index in range(25):
            obs = env.reset(unseen=False, index=index)
            total_reward = 0.

            for i in range(3000):
                image_obs = env.render(camera='sensor.camera.rgb/top')
                action = agent.predict(image_obs, obs)
                obs, reward, done, info = env.step(action[0])

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

if __name__ == '__main__':
    main()
