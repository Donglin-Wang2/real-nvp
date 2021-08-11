import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import Sigmoid
from real_nvp.models.real_nvp import RealNVP


class Reducer(nn.Module):
    def __init__(self, img_shape) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(img_shape), 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Sigmoid()
        )

    def forward(self, x, reverse=False):
        if not reverse:
            return self.encoder(x)
        else:
            return self.decoder(x)


class ContrastiveNVP(nn.Module):
    def __init__(self, img_shape, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super().__init__()
        self.img_shape = img_shape
        self.nvp = RealNVP(num_scales=num_scales, in_channels=in_channels,
                           mid_channels=mid_channels, num_blocks=num_blocks)
        self.tt = Reducer(img_shape)

    def forward(self, x, reverse=False):
        if not reverse:
            z, sldj = self.nvp(x)
            z = torch.flatten(z, start_dim=1).contiguous()
            return self.tt(z), sldj
        else:
            z = self.tt(x, reverse=True)
            z = z.reshape([-1,] + self.img_shape).contiguous()
            return self.nvp(z,reverse=True)