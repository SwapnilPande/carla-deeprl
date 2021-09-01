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

import hydra

from projects.imitation.data_modules import OfflineCarlaDataModule
from projects.imitation.models import ConvAgent, PerceiverAgent, RecurrentAttentionAgent
from projects.reactive_mbrl.create_env import create_env
from projects.reactive_mbrl.ego_model import EgoModel
from projects.reactive_mbrl.agents.pid_agent import PIDAgent
from projects.reactive_mbrl.npc_modeling.kinematic import Kinematic
from projects.reactive_mbrl.data.data_collector import load_ego_model


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
    def on_validation_epoch_end(self, trainer, pl_module):
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
            obs, reward, done, info = self.env.step(np.array([0,-1]))
            # waypoints = self.env.carla_interface.global_planner._waypoints_queue
            # waypoints = np.array(
            #     [
            #         [
            #             w[0].transform.location.x,
            #             w[0].transform.location.y,
            #             w[0].transform.rotation.yaw,
            #         ]
            #         for w in waypoints
            #     ]
            # )
            # model = load_ego_model()
            # npc_predictor = Kinematic(model, waypoints)
            # agent = PIDAgent(npc_predictor)
            # agent.reset(waypoints)

            model.reset()
            for _ in range(self.eval_length):
                image_obs = self.env.render(camera='sensor.camera.rgb/top')

                with torch.no_grad():
                    action = model.predict(image_obs, obs)[0]
                    # action = agent.predict(self.env, info, info['speed'], 8)
                    print(action)

                obs, reward, done, info = self.env.step(action)
                frames.append(image_obs)
                total_reward += reward
                if done:
                    break
            status = info['termination_state']
            success = (status == 'success') or (status == 'none')
            print('success: {}'.format(success))
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

@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg):
    # Loading agent and environment
    if cfg.agent_type == 'conv':
        agent = ConvAgent() # RecurrentAttentionAgent(**cfg.agent) # hydra.utils.instantiate(cfg.algo.agent)
    elif cfg.agent_type == 'attention':
        agent = RecurrentAttentionAgent(**cfg.agent)
    else:
        raise NotImplementedError

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    if cfg.num_eval_episodes > 0:
        output_path = cfg.data['train_dataset']
        env = create_env(cfg.environment, output_path)

        evaluation_callback = EvaluationCallback(env=env, eval_freq=cfg.eval_freq, eval_length=cfg.eval_length, num_eval_episodes=cfg.num_eval_episodes)
        callbacks.append(evaluation_callback)
    else:
        env = None

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
        if env is not None:
            env.close()

    print('Done')


if __name__ == '__main__':
    main()
