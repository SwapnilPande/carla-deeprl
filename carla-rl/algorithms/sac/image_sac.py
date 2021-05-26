from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra

from .actor import DiagGaussianActor
from .critic import DoubleQCritic, Critic
from .utils import to_np, soft_update_params
from agents.torch.models import make_conv_preprocessor
from sac import SAC

from agents.torch.utils import COLOR, CONVERTER

class ImageSAC(SAC):
    """ CQL for mixed observation spaces """
    def __init__(self, *args, **kwargs):
        self.frame_stack = kwargs.pop('frame_stack', 2)
        self.freeze_conv = kwargs.pop('freeze_conv', False)
        self.conv_arch = kwargs.pop('conv_arch', 'vanilla')

        super().__init__(*args, **kwargs)

        self.encoder = make_conv_preprocessor(512, arch=self.conv_arch, frame_stack=self.frame_stack, freeze_conv=self.freeze_conv)
        self.target_encoder = make_conv_preprocessor(512, arch=self.conv_arch, frame_stack=self.frame_stack, freeze_conv=self.freeze_conv)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        if not self.freeze_conv:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters())

    def predict(self, x):
        if len(x.shape) > 1:
            x = x.flatten()
        img, mlp_features = x[:-8], x[-8:]

        num_channels = int(img.size / (112 * 112))
        img = img.reshape(112, 112, num_channels)

        img = torch.FloatTensor(img).permute(2,0,1).cuda() / 255.

        mlp_features = torch.FloatTensor(mlp_features).cuda()
        img_features = self.encoder(img[None]).reshape(1,512)
        state = torch.cat([img_features, mlp_features[None]], dim=1)
        action = self.policy(state).rsample()
        return action.detach().cpu().numpy().reshape(-1, self.action_dim)

    def convert_batch_obs(self, batch):
        (obs, actions, rewards, next_obs, terminals), indices, weights = batch

        img, mlp_features = obs[:,:-8], obs[:,-8:]
        num_channels = int(img.size(1) / (112 * 112))
        img = img.reshape(-1, 112, 112, num_channels)

        img = torch.cuda.FloatTensor(img).permute(0,3,1,2) / 255.
        mlp_features = torch.cuda.FloatTensor(mlp_features)

        next_img, next_mlp_features = next_obs[:,:-8], next_obs[:,-8:]
        next_img = next_img.reshape(-1, 112, 112, num_channels)

        next_img = torch.cuda.FloatTensor(next_img).permute(0,3,1,2) / 255.
        next_mlp_features = torch.cuda.FloatTensor(next_mlp_features)

        img_features = self.encoder(img).reshape(-1,512)
        next_img_features = self.encoder(next_img).reshape(-1,512)

        state = torch.cat([img_features, mlp_features], dim=1)
        next_state = torch.cat([next_img_features, next_mlp_features], dim=1)
        return (state, actions, rewards, next_state, terminals), indices, weights

    def training_step(self, batch, batch_idx, optimizer_idx):
        new_batch = self.convert_batch_obs(batch)
        super().training_step(new_batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        return super().configure_optimizers()

    def set_encoder(self, encoder):
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False


class AsymmetricSAC(SAC):
    """ Image actor, state critic """
    def __init__(self, *args, **kwargs):
        self.frame_stack = kwargs.pop('frame_stack', 2)
        self.freeze_conv = kwargs.pop('freeze_conv', False)
        self.conv_arch = kwargs.pop('conv_arch', 'vanilla')

        super().__init__(*args, **kwargs)

        self.encoder = make_conv_preprocessor(512, arch=self.conv_arch, frame_stack=self.frame_stack, freeze_conv=self.freeze_conv)
        self.target_encoder = make_conv_preprocessor(512, arch=self.conv_arch, frame_stack=self.frame_stack, freeze_conv=self.freeze_conv)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        if not self.freeze_conv:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters())

    def predict(self, x):
        if len(x.shape) > 1:
            x = x.flatten()
        img, mlp_features = x[:-8], x[-8:]

        num_channels = int(img.size / (128 * 128))
        img = img.reshape(128, 128, num_channels)

        img = torch.FloatTensor(img).permute(2,0,1).cuda() / 255.

        mlp_features = torch.FloatTensor(mlp_features).cuda()
        img_features = self.encoder(img[None]).reshape(1,512)
        state = torch.cat([img_features, mlp_features[None]], dim=1)
        action = self.policy(state).rsample()
        return action.detach().cpu().numpy().reshape(-1, self.action_dim)

    def convert_batch_obs(self, batch):
        (obs, actions, rewards, next_obs, terminals), indices, weights = batch

        img, mlp_features = obs[:,:-8], obs[:,-8:]
        num_channels = int(img.size(1) / (128 * 128))
        img = img.reshape(-1, 128, 128, num_channels)

        img = torch.cuda.FloatTensor(img).permute(0,3,1,2) / 255.
        mlp_features = torch.cuda.FloatTensor(mlp_features)

        next_img, next_mlp_features = next_obs[:,:-8], next_obs[:,-8:]
        next_img = next_img.reshape(-1, 128, 128, num_channels)

        next_img = torch.cuda.FloatTensor(next_img).permute(0,3,1,2) / 255.
        next_mlp_features = torch.cuda.FloatTensor(next_mlp_features)

        img_features = self.encoder(img).reshape(-1,512)
        next_img_features = self.encoder(next_img).reshape(-1,512)

        # state = torch.cat([img_features, mlp_features], dim=1)
        # next_state = torch.cat([next_img_features, next_mlp_features], dim=1)
        return ((img_features, mlp_features), actions, rewards, (next_img_features, next_mlp_features), terminals), indices, weights

    def training_step(self, batch, batch_idx, optimizer_idx):
        new_batch = self.convert_batch_obs(batch)
        # super().training_step(new_batch, batch_idx, optimizer_idx)

        self._step += 1
        ((img_features, mlp_features), actions, rewards, (next_img_features, next_mlp_features), terminals), indices, weights = new_batch
        # obs, actions, rewards, next_obs, terminals = batch

        obs = torch.cat([img_features, mlp_features], dim=1)
        next_obs = torch.cat([next_img_features, next_mlp_features], dim=1)

        """
        Policy and Alpha Loss
        """
        # new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(obs)
        dist = self.policy(obs)
        new_obs_actions = dist.rsample()
        log_pi = dist.log_prob(new_obs_actions).sum(-1, keepdim=True)
        
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            # self.manual_backward(alpha_loss, self.alpha_optimizer)
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(mlp_features, new_obs_actions),
            self.qf2(mlp_features, new_obs_actions),
        )

        policy_loss = (alpha.detach()*log_pi - q_new_actions).mean()

        if self._step < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1.forward_factored(mlp_features, actions)
        q2_pred = self.qf2.forward_factored(mlp_features, actions)

        # q1_pred = self.qf1.forward(obs, actions)
        # q2_pred = self.qf2.forward(obs, actions)

        next_dist = self.policy(next_obs)
        new_next_actions = next_dist.rsample()
        new_log_pi = next_dist.log_prob(new_next_actions).sum(-1, keepdim=True)

        if not self.max_q_backup:
            target_q_values = torch.min(
                torch.stack([self.target_qf1.forward_factored(next_mlp_features, new_next_actions),
                self.target_qf2.forward_factored(next_mlp_features, new_next_actions),
                ], dim=1), dim=1).values
            # target_q_values = torch.min(
            #     self.target_qf1.forward(next_obs, new_next_actions),
            #     self.target_qf2.forward(next_obs, new_next_actions),
            # )
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        
        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, 10, self.policy)
            target_qf1_values = self._get_tensor_values(next_mlp_features, next_actions_temp, self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_mlp_features, next_actions_temp, self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        # self.qf1.update(obs, actions, rewards, q_target)
        # self.qf2.update(obs, actions, rewards, q_target)

        trajectory_loss = self.qf_criterion(q1_pred[:,0], q_target[:,0])
        speed_loss = self.qf_criterion(q1_pred[:,1], q_target[:,1])
        light_loss = self.qf_criterion(q1_pred[:,2], q_target[:,2])
        collision_loss = self.qf_criterion(q1_pred[:,3], q_target[:,3])

        self.log('factored/trajectory_loss', trajectory_loss)
        self.log('factored/speed_loss', speed_loss)
        self.log('factored/light_loss', light_loss)
        self.log('factored/collision_loss', collision_loss)

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        # update PER
        td_error_1 = F.mse_loss(q1_pred.flatten(), q_target.flatten(), reduction='none')
        td_error_2 = F.mse_loss(q2_pred.flatten(), q_target.flatten(), reduction='none')
        self._datamodule.dataset.replay_buffer.update_priorities(indices, td_error_1 + td_error_2)

        self.log('qf1/mse_loss', qf1_loss)
        self.log('qf2/mse_loss', qf2_loss)

        if self.use_cql:
            ## add CQL
            random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(self.device)
            curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, self.num_random, self.policy)
            new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, self.num_random, self.policy)
            q1_rand = self._get_tensor_values(obs, random_actions_tensor, self.qf1)
            q2_rand = self._get_tensor_values(obs, random_actions_tensor, self.qf2)
            q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, self.qf1)
            q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, self.qf2)
            q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, self.qf1)
            q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, self.qf2)

            cat_q1 = torch.cat(
                [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
            )
            cat_q2 = torch.cat(
                [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
            )
            std_q1 = torch.std(cat_q1, dim=1)
            std_q2 = torch.std(cat_q2, dim=1)

            if self.min_q_version == 3:
                # importance sampled version
                random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
                cat_q1 = torch.cat(
                    [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
                )

            min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

            if self.with_lagrange:
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 

                self.log('cql/alpha_prime', alpha_prime)
                self.log('cql/alpha_prime_loss', alpha_prime_loss)

                alpha_prime_loss.backward(retain_graph=True)
                # self.manual_backward(alpha_prime_loss, self.alpha_prime_optimizer, retain_graph=True)
                self.alpha_prime_optimizer.step()

            qf1_loss = qf1_loss + min_qf1_loss
            qf2_loss = qf2_loss + min_qf2_loss

            self.log('qf1/cql_loss', min_qf1_loss)
            self.log('qf2/cql_loss', min_qf2_loss)
            self.log('qf1/std', std_q1.mean())
            self.log('qf2/std', std_q2.mean())

        self.log('policy/loss', policy_loss, prog_bar=True)
        self.log('qf1/loss', qf1_loss, prog_bar=True)
        self.log('qf2/loss', qf2_loss, prog_bar=True)
        self.log('qf1/mean', q1_pred.mean())
        self.log('qf2/mean', q2_pred.mean())

        """
        Update networks
        """
        # Update the Q-functions iff 
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        # self.manual_backward(qf1_loss, self.qf1_optimizer, retain_graph=True)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        # self.manual_backward(qf2_loss, self.qf2_optimizer, retain_graph=True)
        self.qf2_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        # self.manual_backward(policy_loss, self.policy_optimizer, retain_graph=True)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )
        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

    def configure_optimizers(self):
        return super().configure_optimizers()

    def set_encoder(self, encoder):
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False


class DBC(SAC):
    """ Uses DBC to learn representations for control using bisimulation metrics
    https://arxiv.org/abs/2006.10742
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dynamics_model = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim + 1)
        )
        self.target_dynamics_model = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim + 1)
        )
        self.target_dynamics_model.load_state_dict(self.dynamics_model.state_dict())
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters())

    def training_step(self, batch, batch_idx, optimizer_idx):
        # with torch.no_grad():
        #     new_batch = self.convert_batch_obs(batch)

        new_batch = batch

        # train policy
        # torch.autograd.set_detect_anomaly(True)
        CQL.training_step(self, new_batch, batch_idx, optimizer_idx)

        # train encoder + dynamics
        curr_z, action, reward, next_z, terminal = new_batch
        batch_size = curr_z.size(0)
        perm = np.random.permutation(batch_size)
        curr_z_2 = curr_z[perm]
        reward_2 = reward[perm]

        curr_z_and_action = torch.cat([curr_z, action], dim=1)
        pred_z_and_reward = self.dynamics_model(curr_z_and_action)
        pred_z, pred_reward = pred_z_and_reward[:,:-1], pred_z_and_reward[:,-1:]
        pred_z_2 = pred_z[perm]

        z_dist = F.smooth_l1_loss(curr_z, curr_z_2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward_2, reduction='none')
        t_dist = F.smooth_l1_loss(pred_z, pred_z_2, reduction='none')

        b_dist = r_dist + t_dist # bisimulation distance
        z_loss = F.mse_loss(b_dist, z_dist)
        self.log('encoder_loss', z_loss)

        pred_loss = F.mse_loss(pred_z, next_z.detach())
        reward_loss = F.mse_loss(pred_reward, reward)
        self.log('pred_loss', pred_loss)
        self.log('reward_loss', reward_loss)

        total_loss = z_loss + pred_loss + reward_loss
        self.log('total_dbc_loss', total_loss)

        self.encoder_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.dynamics_optimizer.step()

        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

        for target_param, param in zip(self.target_dynamics_model.parameters(), self.dynamics_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

    def configure_optimizers(self):
        return super().configure_optimizers() + [self.dynamics_optimizer]
