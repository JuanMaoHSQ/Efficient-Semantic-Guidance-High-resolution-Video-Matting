import torch
from torch import nn

class PFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up4x = nn.Upsample(scale_factor=(4, 4), mode='bilinear', align_corners=True)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(nn.Conv1d(in_channels, 256, 1, 1, bias=False),
                                nn.ReLU(True),
                                nn.Conv1d(256, out_channels, 1, 1, bias=False),
                                nn.Sigmoid())

    def forward_single_frame(self, x, z):
        if z.ndim != 3: 
            z = z.flatten(0, 1)
        z = z.transpose(1, 2)
        z_mp = self.maxpool(z)
        z_att = self.se(z_mp)
        z_att = z_att.unsqueeze(3)
        z_att = z_att.repeat(1,1,1,1)
        x = x * z_att
        return x

    def forward_time_series(self, x, z):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1), z).unflatten(0, (B, T))
        return x

    def forward(self, x, z):
        if x.ndim == 5:
            return self.forward_time_series(x, z)
        else:
            return self.forward_single_frame(x, z)
