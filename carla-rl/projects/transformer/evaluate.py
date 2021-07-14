import os

import numpy as np
import hydra

import torch

from models.decision_transformer import DecisionTransformer
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *


EXPERIMENT_DIR = '/home/brian/temp/2021-06-25_04-10-42/'
CHECKPOINT = 'epoch=0-step=25318.ckpt'



@hydra.main(config_name='{}/.hydra/config.yaml'.format(EXPERIMENT_DIR))
def main(cfg):
    # agent = hydra.utils.instantiate(cfg.algo.agent)
    # agent = SAC.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT), **cfg.algo.agent)
    # agent = agent.cuda().eval()

    # model = DecisionTransformer(8, 2, 128)
    model = DecisionTransformer.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT), state_dim=8, act_dim=2, hidden_size=128)
    model = model.cuda().eval()

    device = 'cuda'

    config = DefaultMainConfig()
    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/topdown'] = {
        'x':13.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'64',
        'sensor_y_res':'64',
        'fov':'90', \
        'sensor_tick': '0.0'}
    scenario_config = NoCrashDenseTown01Config()
    scenario_config.city_name = 'Town02'
    config.populate_config(observation_config=obs_config, scenario_config=scenario_config)

    env_class = CarlaEnv # if not cfg.data_module.use_images else CarlaImageEnv
    env = env_class(config=config, log_dir=os.getcwd())

    state_mean = torch.tensor([0]).to(device=device)
    state_std = torch.tensor([1]).to(device=device)

    ep_returns = []
    statuses = []

    for ep_idx in range(25):

        state = env.reset(unseen=False, index=ep_idx)
        frame = env.render(camera='sensor.camera.rgb/topdown')

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, 8).to(device=device, dtype=torch.float32)
        frames = [frame.copy()] # preprocess_rgb(image)[None].to(device=device)
        actions = torch.zeros((0, 2), device=device, dtype=torch.float32)
        # rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = 100.
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        # timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        # sim_states = []

        episode_return, episode_length = 0, 0
        for t in range(2500):
            # add padding
            actions = torch.cat([actions, torch.zeros((1, 2), device=device)], dim=0)[-25:]
            # rewards = torch.cat([rewards, torch.zeros(1, device=device)])[-25:]

            with torch.no_grad():
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    None,
                    target_return.to(dtype=torch.float32)
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()

            state, reward, done, info = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, 8)
            states = torch.cat([states, cur_state], dim=0)[-25:]

            frame = env.render(camera='sensor.camera.rgb/topdown')
            # image = preprocess_rgb(image).to(device=device)[None]
            # images = torch.cat([images, image], dim=0)[-25:]
            frames.append(frame)

            pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=0)[-25:]

            episode_return += reward
            episode_length += 1

            if done:
                status = info['termination_state']
                print(status)
                break

        ep_returns.append(episode_return)
        statuses.append(status)

    # reward_list = []
    # status_list = []
    # for index in range(25):
    #     obs = env.reset(unseen=False, index=index)
    #     total_reward = 0.

    #     for _ in range(10000):
    #         action = agent.predict(obs)[0]
    #         obs, reward, done, info = env.step(action)

    #         total_reward += reward

    #         # frame = env.render()
    #         # cv2.imshow('frame', frame)
    #         # cv2.waitKey(.01)

    #         if done:
    #             break

    #     reward_list.append(total_reward)
    #     status_list.append(info['termination_state'])
    #     print(status_list[-1])

    #     # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
