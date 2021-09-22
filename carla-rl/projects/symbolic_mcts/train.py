import argparse
import os
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra

from environment import CarlaEnv
from projects.symbolic_mcts.symbolic_env import SymbolicCarlaEnv
from projects.symbolic_mcts.dataset import SymbolicDataset, MLPDataset, symbolic_collate_fn
from projects.symbolic_mcts.models import TransformerAgent, MLPAgent
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *


class EvaluationCallback(Callback):
    def __init__(self, env, eval_freq=5, eval_length=1000, num_eval_episodes=1):
        super().__init__()
        self.env = env
        self.eval_freq = eval_freq
        self.eval_length = eval_length
        self.num_eval_episodes = num_eval_episodes

        self.trainer = None
        self.experiment = None

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.trainer = trainer
        epoch = trainer.current_epoch
        if epoch % self.eval_freq == 0:
            self.evaluate_agent(pl_module)

    def evaluate_agent(self, model):
        epoch = self.trainer.current_epoch
        # checkpoint = os.path.join(os.getcwd(), 'epoch_{}_checkpoint.txt'.format(epoch))
        model.eval()

        rewards = []
        successes = 0
        frames = []

        for index in range(self.num_eval_episodes):
            total_reward = 0.
            obs = self.env.reset(unseen=True, index=index)
            for _ in range(self.eval_length):
                # image_obs = self.env.render(camera='sensor.camera.rgb/top')

                with torch.no_grad():
                    action = model.predict(obs)

                obs, reward, done, info = self.env.step(action)
                # frames.append(image_obs)
                frame = self.env.render(camera='sensor.camera.rgb/front')
                frames.append(frame)
                total_reward += reward
                if done:
                    break

            status = info['termination_state']
            success = (status == 'success')
            successes += int(success)
            rewards.append(total_reward)

        model.log('val/avg_reward', np.mean(rewards))
        model.log('val/num_succeses', successes)

        video_path = os.path.join(os.getcwd(), 'epoch_{}.avi'.format(epoch))
        self.save_video(frames, video_path)

    def save_video(self, frames, fname, fps=15):
        frames = [np.array(frame) for frame in frames]
        height, width = frames[0].shape[0], frames[0].shape[1]
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        out.release()


@hydra.main(config_path='conf/', config_name='train.yaml')
def main(cfg):
    config = DefaultMainConfig()
    config.populate_config(
        observation_config = "VehicleDynamicsObstacleNoCameraConfig",
        action_config = "MergedSpeedScaledTanhConfig",
        reward_config = "Simple2RewardConfig",
        scenario_config = "NoCrashDenseTown02Config",
        testing = False,
        carla_gpu = cfg.gpu
    )

    # add camera + rendering for visualization
    config.obs_config.sensors['sensor.camera.rgb/front'] = {
        'x':2.0,
        'z':1.4,
        'pitch':0.0,
        'sensor_x_res':'128',
        'sensor_y_res':'128',
        'fov':'120',
        'sensor_tick': '0.0'
    }
    config.render_server = True

    if cfg.agent == 'transformer':
        agent = TransformerAgent()
    elif cfg.agent == 'mlp':
        agent = MLPAgent()
    else:
        raise NotImplementedError

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    if cfg.num_eval_episodes > 0:
        env = SymbolicCarlaEnv(config=config)
        evaluation_callback = EvaluationCallback(env=env, eval_freq=cfg.eval_freq, eval_length=cfg.eval_length, num_eval_episodes=cfg.num_eval_episodes)
        callbacks.append(evaluation_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    if cfg.agent == 'transformer':
        dataset = SymbolicDataset(cfg.dataset_path)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=symbolic_collate_fn)
    elif cfg.agent == 'mlp':
        dataset = MLPDataset(cfg.dataset_path)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    else:
        raise NotImplementedError

    try:
        trainer = pl.Trainer(**cfg.trainer, 
            logger=logger,
            callbacks=callbacks,
            max_epochs=cfg.num_epochs)
        trainer.fit(agent, dataloader)
    finally:
        if cfg.num_eval_episodes > 0:
            env.close()

    print('Done')


if __name__ == '__main__':
    main()
