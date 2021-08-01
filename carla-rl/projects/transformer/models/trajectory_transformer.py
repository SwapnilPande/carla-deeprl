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
from .action_binner import *


STATE_TOKENIZER_CHECKPOINT = '/home/scratch/brianyan/outputs/tokenizer/2021-07-27_10-53-00/checkpoints/epoch=34-step=30449.ckpt'
# ACTION_TOKENIZER_CHECKPOINT = '/home/scratch/brianyan/outputs/action_vqvae/2021-07-26_20-00-26/checkpoints/epoch=25-step=7048.ckpt'
VOCAB_SIZE = 256
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
        self.state_tokenizer = VQVAE.load_from_checkpoint(STATE_TOKENIZER_CHECKPOINT, input_dim=8)
        # self.action_tokenizer = VQVAE.load_from_checkpoint(ACTION_TOKENIZER_CHECKPOINT, input_dim=2)

    def forward(self, state_tokens, action_tokens):
        batch_size, seq_length = state_tokens.shape[0], state_tokens.shape[1]-1
        # num_tokens = 2 * (seq_length+1)

        # interleave states and actions
        # tokens = torch.stack([state_tokens, action_tokens], dim=2).view(batch_size, num_tokens)

        # logits = self.transformer(tokens[:,:-1]).logits
        logits = self.transformer(state_tokens).logits

        # loss = F.cross_entropy(logits.reshape(batch_size * (num_tokens-1), VOCAB_SIZE), 
        #     tokens[:,1:].reshape(batch_size * (num_tokens-1)))
        # preds = logits.argmax(dim=-1)
        # accuracy = (preds == tokens[:,1:]).sum() / (batch_size * (num_tokens-1))

        logit = logits.reshape(batch_size, seq_length+1, VOCAB_SIZE)[:,-1]
        action_token = action_tokens[:,-1]

        loss = F.cross_entropy(logit, action_token)
        accuracy = (logit.argmax(dim=-1) == action_token).float().mean()

        return logits, logits.argmax(dim=-1), loss, accuracy

    def get_action(self, states, actions):
        # states = torch.FloatTensor(states).reshape(-1,8)
        # actions = torch.FloatTensor(actions).reshape(-1,2)

        seq_length = states.shape[0]
        num_tokens = 2 * seq_length - 1

        state_tokens = self.state_tokenizer(states).argmax(dim=-1)

        # if len(actions) > 0:
        #     action_tokens = self.action_tokenizer(actions).argmax(dim=-1)
        #     tokens = torch.stack([state_tokens, action_tokens], dim=1).view(1, num_tokens)
        # else:
        #     tokens = state_tokens

        tokens = state_tokens

        # greedy decoding
        logits = self.transformer(tokens).logits
        pred_action_token = logits.argmax(dim=-1)[-1]
        pred_action = decode_action_tokens(pred_action_token[None,None])

        return pred_action

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-4)

    def training_step(self, batch, batch_idx):
        states, actions = batch
        logits, preds, loss, accuracy = self.forward(states, actions)
        self.log('train/loss', loss)
        self.log('train/acc', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        states, actions = batch
        logits, preds, loss, accuracy = self.forward(states, actions)
        self.log('val/loss', loss)
        self.log('val/acc', accuracy)
        return loss
