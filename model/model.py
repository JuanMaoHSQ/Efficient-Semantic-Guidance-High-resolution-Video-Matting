import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .decoder import RecurrentDecoder, Projection
from .dgf import DeepGuidedFilterRefiner
from .fast_guided_filter import FastGuidedFilterRefiner
from .CPF import CPF


class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        # self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
        self.backbone = CPF()
        self.decoder = RecurrentDecoder([16, 24, 48, 128], [80, 40, 32, 16])
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)
        self.refiner = DeepGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        f = self.backbone(src_sm)
        ''''
        torch.Size([1, 12, 16, 256, 256])
        torch.Size([1, 12, 24, 128, 128])
        torch.Size([1, 12, 48, 64, 64])
        torch.Size([1, 12, 128, 32, 32])
        f[9] = z (10, 6, 128)
        '''
        hid, *rec = self.decoder(src_sm, f[9], f[0], f[2], f[4], f[8], r1, r2, r3, r4)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
