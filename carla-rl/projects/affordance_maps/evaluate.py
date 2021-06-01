import os

import numpy as np
import hydra

from agents.torch.sac import SAC
from environment.carla_9_4.env import CarlaEnv


EXPERIMENT_DIR = '/home/brian/temp/factored_dense/2021-04-27_00-57-58'
CHECKPOINT = 'epoch=60-step=60999.ckpt'



@hydra.main(config_name='{}/.hydra/config.yaml'.format(EXPERIMENT_DIR))
def main(cfg):
    # agent = hydra.utils.instantiate(cfg.algo.agent)
    agent = SAC.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT), **cfg.algo.agent)
    agent = agent.cuda().eval()

    reward_list = []
    status_list = []

    env = CarlaEnv(**cfg.environment)
    for index in range(25):
        obs = env.reset(unseen=False, index=index)
        total_reward = 0.

        for _ in range(10000):
            action = agent.predict(obs)[0]
            obs, reward, done, info = env.step(action)

            total_reward += reward

            # frame = env.render()
            # cv2.imshow('frame', frame)
            # cv2.waitKey(.01)

            if done:
                break

        reward_list.append(total_reward)
        status_list.append(info['termination_state'])
        print(status_list[-1])

        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
