import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import resnet18

import pytorch_lightning as pl

import transformers
from transformers import GPT2Model, GPT2LMHeadModel, AutoModelForCausalLM

from .model import TrajectoryModel
from .vqvae import VQVAE


STATE_TOKENIZER_CHECKPOINT = '/home/scratch/brianyan/outputs/tokenizer/2021-07-22_11-13-40/checkpoints/epoch=10-step=18367.ckpt'
VOCAB_SIZE = 128
NUM_EMBEDDINGS = 128

class TrajectoryTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config = transformers.GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_embd=NUM_EMBEDDINGS,
            n_layer=4,
            n_head=8
        )

        self.transformer = GPT2LMHeadModel(config)
        self.state_tokenizer = VQVAE.load_from_checkpoint(STATE_TOKENIZER_CHECKPOINT)

    def forward(self, states):
        batch_size, seq_length = states.shape[0], states.shape[1]-1

        state_tokens, gt_recon = self.state_tokenizer(states, reconstruct=True)
        state_tokens = state_tokens.reshape(batch_size, seq_length+1, -1).argmax(dim=-1)
    
        logits = self.transformer(state_tokens[:,:-1]).logits

        loss = F.cross_entropy(logits.reshape(batch_size * seq_length, VOCAB_SIZE), state_tokens[:,1:].reshape(batch_size * seq_length))
        preds = logits.argmax(dim=-1)
        accuracy = (preds == state_tokens[:,1:]).sum() / (batch_size * seq_length)

        pred_recon = self.state_tokenizer.decode(preds.reshape(-1,1)).reshape(batch_size, seq_length, 8)
        gt_recon_error = F.mse_loss(states, gt_recon)
        pred_recon_error = F.mse_loss(states[:,1:], pred_recon)

        return logits, logits.argmax(dim=-1), loss, accuracy, gt_recon_error, pred_recon_error

    # def get_action(self, states, actions, rewards, returns_to_go, **kwargs):
    #     # we don't care about the past rewards in this model

    #     states = states.reshape(1, -1, self.state_dim)
    #     actions = actions.reshape(1, -1, self.act_dim)
    #     returns_to_go = returns_to_go.reshape(1, -1, 1)
    #     # timesteps = timesteps.reshape(1, -1)

    #     if self.max_length is not None:
    #         states = states[:,-self.max_length:]
    #         actions = actions[:,-self.max_length:]
    #         returns_to_go = returns_to_go[:,-self.max_length:]
    #         # timesteps = timesteps[:,-self.max_length:]

    #         # pad all tokens to sequence length
    #         attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
    #         attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
    #         states = torch.cat(
    #             [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
    #             dim=1).to(dtype=torch.float32)
    #         actions = torch.cat(
    #             [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
    #                          device=actions.device), actions],
    #             dim=1).to(dtype=torch.float32)
    #         returns_to_go = torch.cat(
    #             [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
    #             dim=1).to(dtype=torch.float32)
    #         # timesteps = torch.cat(
    #         #     [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
    #         #     dim=1
    #         # ).to(dtype=torch.long)
    #     else:
    #         attention_mask = None

    #     _, action_preds, return_preds = self.forward(
    #         states, actions, None, returns_to_go, attention_mask=attention_mask, **kwargs)

    #     return action_preds[0,-1]

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-4)

    def training_step(self, batch, batch_idx):
        states, actions = batch
        logits, preds, loss, accuracy, gt_recon_error, pred_recon_error = self.forward(states)
        self.log('train/loss', loss)
        self.log('train/acc', accuracy)
        self.log('train/gt_recon_error', gt_recon_error)
        self.log('train/pred_recon_error', pred_recon_error)
        return loss

    def validation_step(self, batch, batch_idx):
        states, actions = batch
        logits, preds, loss, accuracy, gt_recon_error, pred_recon_error = self.forward(states)
        self.log('val/loss', loss)
        self.log('val/acc', accuracy)
        self.log('val/gt_recon_error', gt_recon_error)
        self.log('val/pred_recon_error', pred_recon_error)
        return loss
