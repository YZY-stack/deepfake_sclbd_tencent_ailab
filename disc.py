import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = sn_double_conv(3, 64)
        self.conv2 = sn_double_conv(64, 128)
        self.conv3 = sn_double_conv(128, 256)
        self.conv4 = sn_double_conv(256, 512)
        [nn.init.xavier_uniform_(
            getattr(self, 'conv{}'.format(i))[j].weight,
            np.sqrt(2)
            ) for i in range(1, 5) for j in range(2)]

        self.l = nn.utils.spectral_norm(nn.Linear(512, 2))
        nn.init.xavier_uniform_(self.l.weight)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.sum(x, [2,3]) # global pool
        out = self.l(x)
        return out
