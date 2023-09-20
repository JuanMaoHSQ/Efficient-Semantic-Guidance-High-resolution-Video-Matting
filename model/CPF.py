import torch
import torch.nn as nn

from .CPF_config import config
from .mobile import Mobile, hswish, MobileDown
from .former import Former
from .bridge import Mobile2Former, Former2Mobile


class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        x_out = self.bn(x_out)
        x_out = self.relu(x_out)
        return [x_out, z_out]


class CPF(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = config['mf294']
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )
        # bneck
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], 3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))

    def forward_single_frame(self, x):
        b, _, _, _ = x.shape
        f = []
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        #z.size() = torch.Size([10, 6, 192])` 
        f.append(x)
        for m in self.block:
            x, z = m([x, z])
            f.append(x)
        f.append(z)
        return f

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
