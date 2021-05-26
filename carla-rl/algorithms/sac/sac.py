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


class SAC(pl.LightningModule):
    """ SAC implementation

    Adapted from https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
    """

    def __init__(
            self,

            obs_dim,
            action_dim,

            actor_cfg,
            critic_cfg,

            discount=0.95,
            reward_scale=1.,

            policy_lr=3e-5,
            qf_lr=3e-4,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,

            # CQL
            use_cql=True,
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            ## sort of backup
            max_q_backup=True,
            deterministic_backup=False,
            num_random=10,
            with_lagrange=True,
            lagrange_thresh=2.0,
            init_log_alpha_prime=-4,

            factored=True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.policy = hydra.utils.instantiate(actor_cfg)
        self.qf1 = hydra.utils.instantiate(critic_cfg)
        self.qf2 = hydra.utils.instantiate(critic_cfg)

        self.target_qf1 = hydra.utils.instantiate(critic_cfg)
        self.target_qf2 = hydra.utils.instantiate(critic_cfg)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod((action_dim,)).item()
            self.register_buffer('log_alpha', torch.zeros(1))
        
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.register_buffer('log_alpha_prime', torch.tensor([float(init_log_alpha_prime)]))

        self.qf_lr = qf_lr
        self.policy_lr = policy_lr

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optim.Adam(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optim.Adam(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.alpha_optimizer = None
        self.alpha_prime_optimizer = None

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        
        self._step = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.use_cql = use_cql
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        self.discrete = False
        self.factored = factored

    def forward(self, x):
        return self.policy(x).mean

    def predict(self, x):
        x = torch.FloatTensor(x).cuda()
        x = x.view(-1, self.obs_dim)
        return self.forward(x).detach().cpu().numpy()

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, policy):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        dist = policy(obs_temp)
        new_obs_actions = dist.rsample()
        new_obs_log_pi = dist.log_prob(new_obs_actions).sum(-1, keepdim=True)
        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self._step += 1
        (obs, actions, rewards, next_obs, terminals), indices, weights = batch
        # obs, actions, rewards, next_obs, terminals = batch

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
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
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

        if self.factored:
            q1_pred = self.qf1.forward_factored(obs, actions)
            q2_pred = self.qf2.forward_factored(obs, actions)
        else:
            q1_pred = self.qf1.forward(obs, actions)
            q2_pred = self.qf2.forward(obs, actions)

        next_dist = self.policy(next_obs)
        new_next_actions = next_dist.rsample()
        new_log_pi = next_dist.log_prob(new_next_actions).sum(-1, keepdim=True)

        if not self.max_q_backup:
            if self.factored:
                target_q_values = torch.min(
                    torch.stack([self.target_qf1.forward_factored(next_obs, new_next_actions),
                    self.target_qf2.forward_factored(next_obs, new_next_actions),
                    ], dim=1), dim=1).values
            else:
                target_q_values = torch.min(
                    self.target_qf1.forward(next_obs, new_next_actions),
                    self.target_qf2.forward(next_obs, new_next_actions),
                )
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        
        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, 10, self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        # self.qf1.update(obs, actions, rewards, q_target)
        # self.qf2.update(obs, actions, rewards, q_target)

        if self.factored:
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
        optimizers = [
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]
        if self.use_automatic_entropy_tuning:
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha.detach()],
                lr=self.policy_lr,
            )
            optimizers.append(self.alpha_optimizer)

        if self.with_lagrange:
            self.log_alpha_prime.requires_grad_()
            self.alpha_prime_optimizer = optim.Adam(
                [self.log_alpha_prime.detach()],
                lr=self.qf_lr,
            )
            optimizers.append(self.alpha_prime_optimizer)
        
        return optimizers

    @property
    def automatic_optimization(self):
        return False
