import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import resnet18

import transformers

from .model import TrajectoryModel
from .trajectory_gpt2 import GPT2Model


class EBMTransformer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=3,
            n_head=1,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        # self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.predict_energy = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # self.embed_image = nn.Sequential(*list(resnet18(pretrained=True).children())[:-3],
        #     nn.Conv2d(256, 8, kernel_size=1, stride=1),
        #     nn.Flatten()
        # )

    def forward(self, x, return_grad=False):
        batch_size = x.shape[0]
        if return_grad:
            x.requires_grad_()
            x.retain_grad()

        with torch.enable_grad():
            hidden_state = self.transformer(inputs_embeds=x)['last_hidden_state']
            hidden_state = hidden_state.reshape(batch_size, -1, self.hidden_size)
            hidden_state = hidden_state.mean(dim=1) # mean pooling
            pred = self.predict_energy(hidden_state)
            if return_grad:
                pred.sum().backward()
                grad = x.grad.detach().clone()
                x.grad.detach_()
                x.grad.zero_()
                return pred, grad
            else:
                return pred

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        states, actions, returns_to_go = batch

        batch_size, seq_length = states.shape[0], states.shape[1]
        timesteps = torch.LongTensor(torch.arange(seq_length)).to(self.device)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        pos_x = stacked_inputs
        neg_x = torch.normal(0, .005, pos_x.shape, requires_grad=True).to(self.device)

        step_size = 10
        for i in range(50):
            _, grad = self.forward(neg_x, return_grad=True)
            neg_x.add_(-step_size * neg_x.grad)

        neg_x = neg_x.detach()

        alpha = 1.
        loss = alpha * (pos_x**2 + neg_x**2) + (pos_x - neg_x)
        loss = loss.mean()
        return loss

    def validation_step(self, batch, batch_idx):
        # return
        states, actions, returns_to_go = batch

        batch_size, seq_length = states.shape[0], states.shape[1]
        timesteps = torch.LongTensor(torch.arange(seq_length)).to(self.device)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        pos_x = stacked_inputs
        neg_x = torch.normal(0, .005, pos_x.shape).to(self.device)

        step_size = 10
        for i in range(50):
            _, grad = self.forward(neg_x, return_grad=True)
            neg_x.add_(-step_size * neg_x.grad)

        neg_x = neg_x.detach()

        pos_energy = self.forward(pos_x)
        neg_energy = self.forward(neg_x)

        self.log('val/pos_energy', pos_energy)
        self.log('val/neg_energy', neg_energy)

        random_x = torch.normal(0, .005, pos_x.shape).to(self.device)
        random_energy = self.forward(random_x)
        self.log('val/random_energy', random_energy)
