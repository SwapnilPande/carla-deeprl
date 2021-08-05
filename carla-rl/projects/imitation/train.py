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

from data_modules import OfflineCarlaDataModule
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *
from models import ConvAgent, PerceiverAgent, AttentionAgent, RecurrentAttentionAgent


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
            model.reset()
            for _ in range(self.eval_length):
                image_obs = self.env.render(camera='sensor.camera.rgb/top')
                action = model.predict(image_obs, obs)[0]
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
    # seed_everything(cfg.seed)

    # Loading agent and environment
    agent = RecurrentAttentionAgent(**cfg.agent) # hydra.utils.instantiate(cfg.algo.agent)

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    if cfg.num_eval_episodes > 0:
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

        env = CarlaEnv(config=config, log_dir=os.getcwd() + '/')

        evaluation_callback = EvaluationCallback(env=env, eval_freq=cfg.eval_freq, eval_length=cfg.eval_length, num_eval_episodes=cfg.num_eval_episodes)
        callbacks.append(evaluation_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    offline_data_module = OfflineCarlaDataModule(cfg.data_module)
    offline_data_module.setup(None)

    try:
        trainer = pl.Trainer(**cfg.trainer, 
            logger=logger,
            callbacks=callbacks,
            max_epochs=cfg.num_epochs)
        trainer.fit(agent, offline_data_module)
    finally:
        env.close()

    print('Done')


if __name__ == '__main__':
    main()
