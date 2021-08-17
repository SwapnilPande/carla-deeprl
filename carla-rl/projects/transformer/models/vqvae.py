import matplotlib.pyplot as plt
import numpy as np

from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import pytorch_lightning as pl


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=8, commitment_cost=.25):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

    def decode(self, ids):
        """ Convert encoding indices to embedding vectors """
        return torch.matmul(ids, self._embedding.weight)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

    def decode(self, ids):
        """ Convert encoding indices to embedding vectors """
        return torch.matmul(ids, self._embedding.weight)


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self._mlp = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            # nn.ReLU(),
            # nn.Linear(256,512)
        )

    def forward(self, inputs):
        # out = inputs.clone()
        # out.requires_grad_()
        # return out
        return self._mlp(inputs)


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        self._mlp = nn.Sequential(
            # nn.Linear(512,256),
            # nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,input_dim)
        )

    def forward(self, inputs):
        # out = inputs.clone()
        # out.requires_grad_()
        # return out
        return self._mlp(inputs)


class VQVAE(pl.LightningModule):
    def __init__(self, input_dim=8):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(input_dim=input_dim)
        self._vq_vae = VectorQuantizerEMA(embedding_dim=128)
        self._decoder = Decoder(input_dim=input_dim)

    def forward(self, x, reconstruct=False):
        z = self._encoder(x)
        loss, quantized, perplexity, ids = self._vq_vae(z)
        if reconstruct:
            recon = self._decoder(quantized)
            return ids, recon
        else:
            return ids

    def decode(self, ids):
        encodings = torch.zeros(ids.shape[0], self._vq_vae._num_embeddings, device=self.device)
        encodings.scatter_(1, ids, 1)
        quantized = self._vq_vae.decode(encodings)
        recon = self._decoder(quantized)
        return recon

    def training_step(self, batch, batch_idx):
        state, action = batch
        z = self._encoder(state)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, state)
        loss = recon_error + vq_loss
        self.log('train/recon_error', recon_error)
        self.log('train/vq_loss', vq_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        state, action = batch
        z = self._encoder(state)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, state)
        loss = recon_error + vq_loss
        self.log('val/recon_error', recon_error)
        self.log('val/vq_loss', vq_loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, amsgrad=False)