
import torch
from torch import nn
from torch import Tensor

class ACF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hamin = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1),
                                   nn.GroupNorm(4, 32),
                                   nn.ELU())
        self.ham1 = nn.Sequential(nn.Conv2d(32, 32, (1, 7), 1, (0, 3)),
                                  nn.GroupNorm(4, 32),
                                  nn.ELU(),
                                  nn.Conv2d(32, 16, (7, 1), 1, (3, 0)),
                                  nn.GroupNorm(2, 16),
                                  nn.ELU())
        self.ham2 = nn.Sequential(nn.Conv2d(32, 32, (7, 1), 1, (3, 0)),
                                  nn.GroupNorm(4, 32),
                                  nn.ELU(),
                                  nn.Conv2d(32, 16, (1, 7), 1, (0, 3)),
                                  nn.GroupNorm(2, 16),
                                  nn.ELU())
        self.hamse = nn.Sequential(nn.Conv2d(32, out_channels, 1, 1, bias=True),
                                   nn.Sigmoid())

    def forward_single_frame(self, x, a):
        a_o = self.hamin(a)
        a1 = self.ham1(a_o)
        a2 = self.ham2(a_o)
        a_se = torch.cat([a1, a2], 1)
        a_se1 = self.hamse(a_se)
        out = a_se1 * x
        return out

    def forward_time_series(self, x, a):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x

    def forward(self, x, a):
        if x.ndim == 5:
            return self.forward_time_series(x, a)
        else:
            return self.forward_single_frame(x, a)

