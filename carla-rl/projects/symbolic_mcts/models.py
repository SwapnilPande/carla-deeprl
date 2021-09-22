import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from transformers import BertConfig, BertModel


def positional_encoding(p, L=10):
    return torch.stack([torch.sin(2**i * np.pi * p) for i in range(L)] + [torch.cos(2**i * np.pi * p) for i in range(L)], dim=-1)


class TransformerAgent(pl.LightningModule):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size

        config = BertConfig(
            vocab_size=1, # we do our own embeddings
            num_attention_heads=8,
            hidden_size=self.embedding_size,
            intermediate_size=512,
        )
        self.model = BertModel(config)

        self.action_predictor = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

        self.segment_embedding = nn.Embedding(2, embedding_size)

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

        self.waypoint_encoder = nn.Linear(1, 128)

    def make_ego_token(self, obs_dict):
        ego_features = obs_dict['ego_features']
        ego_vehicle_token = self.make_vehicle_token(ego_features)
        ego_encodings = self.ego_encoder(
            torch.FloatTensor([obs_dict['light'], obs_dict['next_orientation'], obs_dict['dist_to_trajectory']]).cuda()
        )
        ego_token = ego_vehicle_token + ego_encodings
        return ego_token

    def make_vehicle_token(self, actor_features, is_ego=False):
        positions = torch.tensor(actor_features['bounding_box'])[::2] / 20.
        positional_encodings = positional_encoding(positions, L=4).cuda().reshape(-1,16).repeat(1,8)

        feature_encodings = self.vehicle_encoder(
            torch.FloatTensor([actor_features['theta'], actor_features['speed']]).cuda()
        )

        segment_encodings = self.segment_embedding(torch.tensor([0]).cuda())

        token = feature_encodings + positional_encodings + segment_encodings
        return token

    def make_waypoint_tokens(self, waypoints):
        waypoints = torch.tensor(waypoints)[:,:2] / 20.
        positional_encodings = positional_encoding(waypoints, L=4).cuda().reshape(-1,16).repeat(1,8)

        feature_encodings = self.waypoint_encoder(
            torch.FloatTensor([i for i in range(len(waypoints))]).reshape(-1,1).cuda()
        )

        segment_encodings = self.segment_embedding(torch.tensor([1]).cuda())
    
        token = feature_encodings + positional_encodings + segment_encodings
        return token

    def forward(self, obs_batch):
        batch_size = len(obs_batch)
        token_batch = []
        for i, obs in enumerate(obs_batch):
            ego_token = self.make_ego_token(obs)
            if obs['vehicle_features']:
                vehicle_tokens = torch.stack([self.make_vehicle_token(obs['vehicle_features'][idx]) for idx in obs['vehicle_features']]).reshape(-1,self.embedding_size)
            else:
                vehicle_tokens = torch.zeros((0,self.embedding_size)).cuda()
            waypoint_tokens = self.make_waypoint_tokens(obs['next_waypoints'])
            token_seq = torch.cat([ego_token, vehicle_tokens, waypoint_tokens], dim=0)
            token_batch.append(token_seq)

        if batch_size > 1:
            # need to pad each sequence to the max token sequence length
            max_num_tokens = max([len(seq) for seq in token_batch])
            masks = torch.zeros((batch_size, max_num_tokens))
            for i, token_seq in enumerate(token_batch):
                curr_num_tokens = token_batch[i].shape[0]
                padding = torch.zeros((max_num_tokens-curr_num_tokens, self.embedding_size)).cuda()
                token_batch[i] = torch.cat([token_seq, padding], dim=0)
                masks[i] = torch.cat([torch.ones(curr_num_tokens), torch.zeros(max_num_tokens-curr_num_tokens)])

            token_batch = torch.stack(token_batch)
            
            output = self.model(
                inputs_embeds=token_batch.float(),
                attention_mask=masks.cuda().float(),
            )
        else:
            token_batch = torch.stack(token_batch)
            output = self.model(
                inputs_embeds=token_batch.float()
            )

        hidden_state = output[0][:,0] # hidden state of ego token only
        pred_action = self.action_predictor(hidden_state)

        return pred_action

    def predict(self, obs):
        action = self.forward([obs])
        return action.detach().cpu().numpy().reshape(2)

    def training_step(self, batch, batch_idx):
        obs, action, reward, done, next_obs = batch
        pred_action = self.forward(obs)
        loss = F.mse_loss(pred_action, action)
        self.log('train/bc_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, action, reward, done, next_obs = batch
        pred_action = self.forward(obs)
        loss = F.mse_loss(pred_action, action)
        self.log('val/bc_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)


class MLPAgent(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.mlp_extractor = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, obs):
        feature = self.mlp_extractor(obs)
        pred_action = self.action_predictor(feature)
        return pred_action

    def predict(self, obs_dict):
        obs = torch.FloatTensor([
            obs_dict['next_orientation'],
            obs_dict['dist_to_trajectory'],
            obs_dict['ego_features']['speed'],
            obs_dict['light'],
            obs_dict['obstacle_dist'],
            obs_dict['obstacle_speed']
        ]).cuda()[None]
        action = self.forward(obs)
        return action.detach().cpu().numpy().reshape(2)

    def training_step(self, batch, batch_idx):
        obs, action, reward, done, next_obs = batch
        pred_action = self.forward(obs)
        loss = F.mse_loss(pred_action, action)
        self.log('train/bc_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, action, reward, done, next_obs = batch
        pred_action = self.forward(obs)
        loss = F.mse_loss(pred_action, action)
        self.log('val/bc_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
