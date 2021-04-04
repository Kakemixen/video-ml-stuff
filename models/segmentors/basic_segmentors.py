import torch
from torch import nn as nn

from models.blocks.conv_blocks import ConvBNReLU

class TrivialSegmentor(nn.Module):
    def __init__(self, in_c=32, dec_c=16, out_c=1, upscale_x=1):
        super().__init__()
        blocks = [ ConvBNReLU(in_c, dec_c),
                   nn.UpsamplingBilinear2d(scale_factor=2) ]
        for _ in range(1, upscale_x):
            blocks.append(ConvBNReLU(dec_c, dec_c))
            blocks.append(nn.UpsamplingBilinear2d(scale_factor=2))
        blocks.append(ConvBNReLU(dec_c, out_c, ks=1, s=1, p=0, 
            batch_norm=False, activation=False))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

