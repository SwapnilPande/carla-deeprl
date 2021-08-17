import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
import pytorch_lightning as pl
import numpy as np

from common.utils import preprocess_rgb
from perceiver_pytorch import Perceiver


class ConvAgent(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        self.measurement_mlp = nn.Linear(8, 2048)
        self.action_mlp = nn.Sequential(
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self, image, mlp_features):
        image = self.conv(image)
        image = image.view(image.shape[0], -1)
        mlp_features = mlp_features.reshape(-1,8)
        mlp_features = self.measurement_mlp(mlp_features)
        features = image + mlp_features
        action = self.action_mlp(features)
        return action

    def reset(self):
        pass

    def predict(self, image, mlp_features):
        image = preprocess_rgb(image, image_size=(64,64)).cuda()[None]
        mlp_features = torch.FloatTensor(mlp_features).cuda().reshape(1,-1)
        out = self.forward(image, mlp_features)
        action = out.detach().cpu().numpy()
        return np.clip(action, -1, 1)

    def training_step(self, batch, batch_idx):
        (image, mlp_features), action = batch

        batch_size = image.shape[0]
        H = image.shape[1]

        image = image.reshape(batch_size,H,3,64,64)
        mlp_features = mlp_features.reshape(batch_size,H,8)
        action = action.reshape(batch_size,H,2)

        loss = 0.0
        for t in range(H):
            pred_action = self.forward(image[:,t], mlp_features[:,t])
            loss += F.mse_loss(pred_action, action[:,t])

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, mlp_features), action = batch

        batch_size = image.shape[0]
        H = image.shape[1]

        image = image.reshape(batch_size,H,3,64,64)
        mlp_features = mlp_features.reshape(batch_size,H,8)
        action = action.reshape(batch_size,H,2)

        loss = 0.0
        for t in range(H):
            pred_action = self.forward(image[:,t], mlp_features[:,t])
            loss += F.mse_loss(pred_action, action[:,t])

        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class PerceiverAgent(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.perceiver = Perceiver(
            input_channels = 3,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = 6,                   # depth of net
            num_latents = 128,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 256,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,
            latent_dim_head = 64,
            num_classes = 512,          # output number of classes
            attn_dropout = 0.,
            ff_dropout = 0.,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )
        self.measurement_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128,512)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        
    def forward(self, image, mlp_features):
        image = image.permute(0,2,3,1)
        image = self.perceiver(image)
        image = image.view(image.shape[0], -1)
        mlp_features = mlp_features.reshape(-1,8)
        mlp_features = self.measurement_mlp(mlp_features)
        features = torch.cat([image, mlp_features], dim=1)
        action = self.action_mlp(features)
        return action

    def predict(self, image, mlp_features):
        image = preprocess_rgb(image).cuda()[None]
        mlp_features = torch.FloatTensor(mlp_features).cuda().reshape(1,-1)
        out = self.forward(image, mlp_features)
        action = out.detach().cpu().numpy()
        return np.clip(action, -1, 1)

    def training_step(self, batch, batch_idx):
        (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal = batch
        pred_action = self.forward(image, mlp_features)
        loss = F.mse_loss(pred_action, action)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal = batch
        pred_action = self.forward(image, mlp_features)
        loss = F.mse_loss(pred_action, action)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class SpatialBasis:
    """
    Source: https://github.com/cjlovering/Towards-Interpretable-Reinforcement-Learning-Using-Attention-Augmented-Agents-Replication/blob/master/attention.py
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=28, width=28, channels=64):
        h, w, d = height, width, channels

        p_h = torch.mul(torch.arange(1, h+1).unsqueeze(1).float(), torch.ones(1, w).float()) * (np.pi / h)
        p_w = torch.mul(torch.ones(h, 1).float(), torch.arange(1, w+1).unsqueeze(0).float()) * (np.pi / w)
        
        # NOTE: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values. Still, I think what I have is aligned with what
        # they did, but I am less confident in this step.
        U = V = 4 # size of U, V. 
        u_basis = v_basis = torch.arange(1, U+1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum('hwu,hwv->hwuv', torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self.S = out

    def __call__(self, X):
        # Stack the spatial bias (for each batch) and concat to the input.
        batch_size = X.size()[0]
        S = torch.stack([self.S] * batch_size).to(X.device)
        return torch.cat([X, S], dim=3)


def spatial_softmax(A):
    # A: batch_size x h x w x d
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (not over queries)
    A = A.reshape(b, h * w, d)
    A = F.softmax(A, dim=1)
    # Reshape A to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    # TODO: Check this function again.
    b, h, w, c = A.size()
    A = A.reshape(b, h * w, c).transpose(1, 2)

    _, _, _, d = V.size()
    V = V.reshape(b, h * w, d)

    return torch.matmul(A, V)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """Initialize stateful ConvLSTM cell.
        
        Parameters
        ----------
        input_channels : ``int``
            Number of channels of input tensor.
        hidden_channels : ``int``
            Number of channels of hidden state.
        kernel_size : ``int``
            Size of the convolutional kernel.
            
        Paper
        -----
        https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
        
        Referenced code
        ---------------
        https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py        
        https://github.com/cjlovering/Towards-Interpretable-Reinforcement-Learning-Using-Attention-Augmented-Agents-Replication/blob/master/attention.py
        """
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

        self.prev_hidden = None

    def forward(self, x):
        if self.prev_hidden is None:
            batch_size, _, height, width = x.size()
            h, c = self.init_hidden(
                batch_size, self.hidden_channels, height, width, x.device
            )
        else:
            h, c = self.prev_hidden

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)

        self.prev_hidden = ch, cc
        return ch, cc   

    def reset(self):
        del self.prev_hidden
        self.prev_hidden = None

    def init_hidden(self, batch_size, hidden, height, width, device):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
        return (
            torch.zeros(batch_size, hidden, height, width, requires_grad=True).to(
                device
            ),
            torch.zeros(batch_size, hidden, height, width, requires_grad=True).to(
                device
            ),
        )


class RecurrentAttentionAgent(pl.LightningModule):
    def __init__(self, c_v=48, c_k=16, c_s=16, H=20, conv_type='convlstm', attention_type='bottom_up'):
        super().__init__()
        self.c_v = c_v
        self.c_k = c_k
        assert c_v + c_k == 64, 'Invalid c_v, c_k values'

        self.c_s = c_s

        self.conv = nn.Sequential(*list(resnet18(pretrained=True).children())[:-5])
        self.spatial_basis = SpatialBasis(height=16, width=16, channels=self.c_s)
        self.answer_processor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.measurement_mlp = nn.Linear(8, 2048)

        self.conv_type = conv_type
        assert conv_type in (
            'convlstm',         # ConvLSTM, with recurrence
            'conv'              # ResNet, no recurrence
        )

        if conv_type == 'convlstm':
            self.conv_lstm = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
            self.convh = None
        elif conv_type == 'conv':
            pass
        else:
            raise NotImplementedError

        self.attention_type = attention_type
        assert attention_type in (
            'bottom_up',        # query generated from image
            'lstm',             # query generated by LSTM top-down
            'transformer',       # query generated by transformer top-down
            'fixed_learned',    # fixed query, learnable
            'fixed_random',     # fixed random query, not learned
            'random'            # every query is random
        ), 'Invalid attention_type, got {}'.format(attention_type)

        if attention_type == 'fixed_random':
            self.query = torch.normal(0,1,(1,32,self.c_k+self.c_s),requires_grad=False)
        elif attention_type == 'fixed_learned':
            self.query = nn.Parameter(torch.normal(0,1,(1,32,self.c_k+self.c_s),requires_grad=True))
        elif attention_type == 'bottom_up':
            self.query_network = nn.Sequential(
                *list(resnet18(pretrained=True).children())[:-5],
                nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 32 * (self.c_k+self.c_s))
            )
        elif attention_type == 'lstm':
            self.query_network = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 32 * (self.c_k+self.c_s))
            )
            self.lstm = nn.LSTMCell(128,128)
            self.prev_out = None
            self.h = None
        elif attention_type == 'transformer':
            pass
        elif attention_type == 'random':
            pass
        else:
            raise NotImplementedError

    def reset(self):
        """ Resets LSTM hidden states """

        if self.conv_type == 'convlstm':
            self.conv_lstm.reset()
            if self.convh is not None:
                del self.convh
                self.convh = None

        if self.attention_type == 'lstm':
            if self.prev_out is not None:
                del self.prev_out
                self.prev_out = None
            if self.h is not None:
                del self.h
                self.h = None

    def forward(self, image, mlp_features, return_map=False):
        batch_size = image.shape[0]
        x = self.conv(image)
        if self.conv_type == 'convlstm':
            x, self.convh = self.conv_lstm(x)

        x = x.permute(0,2,3,1) # move channels to the end
        k, v = x[...,:self.c_k], x[...,self.c_k:] # split into key-value along channel
        k, v = self.spatial_basis(k), self.spatial_basis(v)

        if self.attention_type in ('fixed_learned', 'fixed_random'):
            q = self.query.clone().to(self.device)
            q = q.repeat(batch_size, 1, 1)
        elif self.attention_type == 'random':
            q = torch.normal(0,1,(batch_size,32,self.c_k+self.c_s)).to(self.device)
        elif self.attention_type == 'bottom_up':
            q = self.query_network(image)
            q = q.reshape(batch_size, 32, self.c_k + self.c_s)
        elif self.attention_type == 'lstm':
            if self.prev_out is None:
                self.prev_out = torch.zeros((batch_size,128)).to(self.device)
            q = self.query_network(self.prev_out)
            q = q.reshape(batch_size, 32, self.c_k + self.c_s)
        else:
            raise NotImplementedError

        A = torch.matmul(k, q.transpose(2,1).unsqueeze(1))
        A = spatial_softmax(A)
        if return_map:
            attention_map = A.clone().detach()
        a = apply_alpha(A, v)
        a = a.reshape(-1, 2048)
        mlp_features = mlp_features.reshape(-1,8)
        mlp_features = self.measurement_mlp(mlp_features)
        # out = torch.cat([a, mlp_features], dim=1)
        out = a + mlp_features
        out = self.answer_processor(out)

        if self.attention_type == 'lstm':
            if self.h is None:
                self.h, c = self.lstm(out)
            else:
                self.h, c = self.lstm(out, (self.prev_out, self.h))
            self.prev_out = out

        if return_map:
            return self.action_predictor(out), attention_map
        else:
            return self.action_predictor(out)

    def predict(self, image, mlp_features, return_map=False):
        image = preprocess_rgb(image, image_size=(64,64)).cuda()[None]
        mlp_features = torch.FloatTensor(mlp_features).cuda().reshape(1,-1)
        if return_map:
            out, attention_map = self.forward(image, mlp_features, return_map=return_map)
            action = out.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            return action, attention_map.detach().cpu().numpy()
        else:
            out = self.forward(image, mlp_features)
            action = out.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            return action

    def training_step(self, batch, batch_idx):
        (image, mlp_features), action = batch

        batch_size = image.shape[0]
        H = image.shape[1]

        image = image.reshape(batch_size,H,3,64,64)
        mlp_features = mlp_features.reshape(batch_size,H,8)
        action = action.reshape(batch_size,H,2)

        self.reset()

        loss = 0.0
        for t in range(H):
            pred_action = self.forward(image[:,t], mlp_features[:,t])
            loss += F.mse_loss(pred_action, action[:,t])

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, mlp_features), action = batch

        batch_size = image.shape[0]
        H = image.shape[1]

        image = image.reshape(batch_size,H,3,64,64)
        mlp_features = mlp_features.reshape(batch_size,H,8)
        action = action.reshape(batch_size,H,2)

        self.reset()

        loss = 0.0
        for t in range(H):
            pred_action = self.forward(image[:,t], mlp_features[:,t])
            loss += F.mse_loss(pred_action, action[:,t])

        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-5)
