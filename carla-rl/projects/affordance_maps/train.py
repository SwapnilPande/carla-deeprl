import argparse
import os
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from omegaconf import DictConfig, OmegaConf
import hydra

from common.data_modules import OfflineCarlaDataModule, OnlineCarlaDataModule
from environment import CarlaEnv
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
    def on_epoch_end(self, trainer, pl_module):
        self.trainer = trainer
        epoch = trainer.current_epoch
        if epoch % self.eval_freq == 0:
            self.evaluate_agent(pl_module)

    def evaluate_agent(self, model):
        epoch = self.trainer.current_epoch
        checkpoint = os.path.join(os.getcwd(), 'epoch_{}_checkpoint.txt'.format(epoch))
        model.eval()

        rewards = []
        scores = []
        successes = 0
        frames = []

        for index in range(self.num_eval_episodes):
            total_reward = 0.
            obs = self.env.reset(unseen=True, index=index)
            for _ in range(self.eval_length):
                action = model.predict(obs)[0]
                obs, reward, done, info = self.env.step(action)
                frames.append(self.env.render(camera='sensor.camera.rgb/top'))
                total_reward += reward
                if done:
                    break
            status = info['termination_state']
            success = (status == 'success') or (status == 'none')
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

@hydra.main(config_path='conf', config_name='train.yaml')
def main(cfg):
    # For reproducibility
    seed_everything(cfg.seed)

    # Loading agent and environment
    agent = hydra.utils.instantiate(cfg.algo.agent)

    config = DefaultMainConfig()
    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/top'] = {
        'x':13.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'64',
        'sensor_y_res':'64',
        'fov':'90', \
        'sensor_tick': '0.0'}
    scenario_config = NoCrashDenseTown02Config()
    action_config = MergedSpeedScaledTanhConfig()

    config.populate_config(observation_config=obs_config, scenario_config=scenario_config)
    env = CarlaEnv(config=config)

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    evaluation_callback = EvaluationCallback(env=env, eval_freq=cfg.eval_freq, eval_length=cfg.eval_length, num_eval_episodes=cfg.num_eval_episodes)
    callbacks.append(evaluation_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    offline_data_module = OfflineCarlaDataModule(cfg.data_module)

    try:
        # Offline training
        if cfg.train_offline:
            trainer = pl.Trainer(**cfg.trainer, 
                logger=logger,
                callbacks=callbacks,
                max_epochs=cfg.offline_epochs)
            trainer.fit(agent, offline_data_module)


        # Online training
        if cfg.train_online:
            online_data_module = OnlineCarlaDataModule(agent, env, cfg.data_module)
            agent._datamodule = online_data_module
            online_data_module.populate(cfg.data_module.populate_size) # populate buffer with offline data
            trainer = pl.Trainer(**cfg.trainer,
                logger=logger,
                callbacks=callbacks,
                max_epochs=cfg.online_epochs)
            if cfg.train_offline:
                trainer.current_epoch = cfg.offline_epochs
                
            trainer.fit(agent, online_data_module)
    finally:
        env.close()

    print('Done')


if __name__ == '__main__':
    main()
