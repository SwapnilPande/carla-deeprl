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

# from leaderboard.utils.statistics_manager import StatisticsManager

from omegaconf import DictConfig, OmegaConf
import hydra

# from stable_baselines.common.vec_env import DummyVecEnv
# from agents.tf.ppo import PPO

from data_modules import TransformerDataModule, TokenizedDataModule
# from algorithms.bc import BC, ImageBC
# from algorithms.sac import SAC, ImageSAC
from models.decision_transformer import DecisionTransformer
from models.trajectory_transformer import TrajectoryTransformer
from models.ebm import EBMTransformer
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *
from utils import preprocess_rgb, preprocess_topdown


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
        device = model.device

        state = self.env.reset()
        frame = self.env.render(camera='sensor.camera.rgb/topdown')

        states = torch.from_numpy(state).reshape(1, 8).to(device=device, dtype=torch.float32)
        frames = [frame.copy()]
        actions = torch.zeros((0, 2), device=device, dtype=torch.float32)

        episode_return, episode_length = 0, 0
        for t in range(self.eval_length):
            # actions = torch.cat([actions, torch.zeros((1, 2), device=device)], dim=0)[-25:]

            with torch.no_grad():
                action = model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                )
                # actions[-1] = action
                action = action.detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, 8)
            states = torch.cat([states, cur_state], dim=0)[-25:]

            frame = self.env.render(camera='sensor.camera.rgb/topdown')
            frames.append(frame)

            # pred_return = target_return[0,-1]
            # target_return = torch.cat(
            #     [target_return, pred_return.reshape(1, 1)], dim=0)[-25:]

            episode_return += reward
            episode_length += 1

            print(episode_length, action)

            if done:
                break

        self.log('val/episode_return', episode_return)

        video_path = os.path.join(os.getcwd(), 'epoch_{}.avi'.format(epoch))
        self.save_video(frames, video_path)

    def save_video(self, frames, fname, fps=15):
        frames = [np.array(frame) for frame in frames]
        height, width = frames[0].shape[0], frames[0].shape[1]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        out.release()

@hydra.main(config_path='../affordance_maps/conf', config_name='train.yaml')
def main(cfg):
    # For reproducibility
    # seed_everything(cfg.seed)

    agent = TrajectoryTransformer()

    config = DefaultMainConfig()

    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/topdown'] = {
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

    try:
        # Setting up logger and checkpoint/eval callbacks
        logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
        callbacks = []

        evaluation_callback = EvaluationCallback(env, eval_freq=1, eval_length=1000)
        callbacks.append(evaluation_callback)

        checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
        callbacks.append(checkpoint_callback)

        cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

        offline_data_module = TokenizedDataModule('/zfsauton/datasets/ArgoRL/brianyan/carla_dataset/')
        offline_data_module.setup(None)

        trainer = pl.Trainer(**cfg.trainer, 
            logger=logger,
            callbacks=callbacks,
            max_epochs=cfg.offline_epochs)
        trainer.fit(agent, offline_data_module)

    finally:
        env.close()
        print('Done')


if __name__ == '__main__':
    main()
