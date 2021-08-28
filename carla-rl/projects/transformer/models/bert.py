"""
TODO checklist
    - tokenizers / embeddings
        - image encoder
        - symbol encoder
        - positional / type embeddings
    - transformer model (BERT)
    - pretraining tasks
        - multi-modal masked modeling
        - next sentence prediction (?)
        - CLIP
    - behavior cloning training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
import pytorch_lightning as pl

from transformers import BertConfig, BertModel
from common.utils import preprocess_rgb


class VisualConvEncoder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size

        self.conv = nn.Sequential(
            *list(resnet18(pretrained=True).children())[:-2],
            nn.AvgPool2d(2,stride=2),
            nn.Flatten(),
            nn.Linear(2048, self.embedding_size)
        )

    def forward(self, x):
        return self.conv(x)


class SymbolicEncoder(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()

        self.embedding_size = embedding_size

        self.mlp_dict = nn.ModuleDict({
            'route': nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_size)
            ),
            'ego': nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_size)
            ),
            'obstacle': nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_size)
            ),
            'light': nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_size)
            )
        })

    def forward(self, x):
        return {k: self.mlp_dict[k](x[k]) for k in x}


class VisualAttentionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        raise NotImplementedError


class VisualSymbolicBert(pl.LightningModule):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size

        self.image_encoder = VisualConvEncoder(embedding_size=embedding_size)
        self.symbol_encoder = SymbolicEncoder(embedding_size=embedding_size)

        self.position_embeddings = nn.Embedding(5, embedding_size)

        config = BertConfig(
            vocab_size=1, # we do our own embeddings
            num_attention_heads=8,
            hidden_size=self.embedding_size,
            intermediate_size=512,
        )
        self.transformer = BertModel(config)

        self.action_predictor = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, x):
        image, symbol_features = x
        batch_size = image.shape[0]

        image_embeds = self.image_encoder(image).unsqueeze(1) # TODO: handle multiple image patch tokens
        symbol_embeds = self.symbol_encoder(symbol_features)
        symbol_embeds = torch.stack([symbol_embeds[k] for k in symbol_embeds], dim=1)
        input_embeds = torch.cat([image_embeds, symbol_embeds], dim=1)

        # position_ids = torch.LongTensor(range(input_embeds.shape[1]))[None].repeat(batch_size,1).to(self.device)
        # position_embeds = self.position_embeddings(position_ids)
        # input_embeds += position_embeds

        outputs = self.transformer(
            inputs_embeds=input_embeds
        )
        pooler_output = outputs[1]

        pred_action = self.action_predictor(pooler_output)
        return pred_action

    def predict(self, image, symbolic_features):
        image = preprocess_rgb(image, image_size=(128,128)).cuda()[None]
        symbolic_features = torch.FloatTensor(symbolic_features).cuda().reshape(1,-1)
        symbolic_features = {
            'route': symbolic_features[:,[0,5]],
            'ego': symbolic_features[:,[3,4]],
            'obstacle': symbolic_features[:,[1,2]],
            'light': symbolic_features[:,[7]]
        }
        out = self.forward((image, symbolic_features))
        action = out.detach().cpu().numpy()
        return action

    def training_step(self, batch, batch_idx):
        (image, symbolic_features), action = batch
        pred_action = self.forward((image, symbolic_features))
        loss = F.mse_loss(action, pred_action)
        self.log('train/bc_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, symbolic_features), action = batch
        pred_action = self.forward((image, symbolic_features))
        loss = F.mse_loss(action, pred_action)
        self.log('val/bc_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)


if __name__ == '__main__':
    model = VisualSymbolicBert()

    BATCH_SIZE = 32
    image = torch.rand(BATCH_SIZE,3,128,128)
    symbolic_features = {
        'route': torch.rand(BATCH_SIZE,2),
        'ego': torch.rand(BATCH_SIZE,2),
        'obstacle': torch.rand(BATCH_SIZE,2),
        'light': torch.rand(BATCH_SIZE,1)
    }
    pred_action = model((image, symbolic_features))
    print(pred_action)