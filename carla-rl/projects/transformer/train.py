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

from data_modules import TransformerDataModule
# from algorithms.bc import BC, ImageBC
# from algorithms.sac import SAC, ImageSAC
from models.decision_transformer import DecisionTransformer
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

        state_mean = torch.tensor([0]).to(device=device)
        state_std = torch.tensor([1]).to(device=device)

        state = self.env.reset()
        frame = self.env.render(camera='sensor.camera.rgb/topdown')

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
        for t in range(self.eval_length):

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

            state, reward, done, _ = self.env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, 8)
            states = torch.cat([states, cur_state], dim=0)[-25:]

            frame = self.env.render(camera='sensor.camera.rgb/topdown')
            # image = preprocess_rgb(image).to(device=device)[None]
            # images = torch.cat([images, image], dim=0)[-25:]
            frames.append(frame)

            pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=0)[-25:]

            episode_return += reward
            episode_length += 1

            print(episode_length, reward)

            if done:
                break

        # print(episode_return, episode_length)
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

    agent = DecisionTransformer(8, 2, 128)

    config = DefaultMainConfig()
    config.server_fps = 20

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

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(observation_config=obs_config, scenario_config=scenario_config, action_config=action_config)

    env_class = CarlaEnv # if not cfg.data_module.use_images else CarlaImageEnv
    env = env_class(config=config, log_dir=os.getcwd())

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    evaluation_callback = EvaluationCallback(env=env, eval_freq=cfg.eval_freq, eval_length=cfg.eval_length, num_eval_episodes=cfg.num_eval_episodes)
    callbacks.append(evaluation_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    offline_data_module = TransformerDataModule(['/zfsauton/datasets/ArgoRL/brianyan/expert_data/'])
    offline_data_module.setup(None)

    try:
        # Offline training
        if cfg.train_offline:
            trainer = pl.Trainer(**cfg.trainer, 
                logger=logger,
                callbacks=callbacks,
                max_epochs=cfg.offline_epochs)
            trainer.fit(agent, offline_data_module)


        # # Online training
        # if cfg.train_online:
        #     online_data_module = OnlineCarlaDataModule(agent, env, cfg.data_module)
        #     agent._datamodule = online_data_module
        #     online_data_module.populate(cfg.data_module.populate_size) # populate buffer with offline data
        #     trainer = pl.Trainer(**cfg.trainer,
        #         logger=logger,
        #         callbacks=callbacks,
        #         max_epochs=cfg.online_epochs)
        #     if cfg.train_offline:
        #         trainer.current_epoch = cfg.offline_epochs
                
        #     trainer.fit(agent, online_data_module)
    finally:
        env.close()

    print('Done')


if __name__ == '__main__':
    main()
