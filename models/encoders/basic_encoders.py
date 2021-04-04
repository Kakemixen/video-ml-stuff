import torch
from torch import nn as nn

from models.blocks.conv_blocks import ConvBNReLU

class TrivialEncoder(nn.Module):
    def __init__(self, in_c=3, enc_c=16, out_c=32, downscale_x=1):  
        super().__init__()
        blocks = [ConvBNReLU(in_c, enc_c, s=2)]
        blocks.append(ConvBNReLU(enc_c, enc_c))
        for _ in range(1, downscale_x):
            blocks.append(ConvBNReLU(enc_c, enc_c, s=2))
            blocks.append(ConvBNReLU(enc_c, enc_c))
        blocks.append(ConvBNReLU(enc_c, out_c, ks=1, s=1, p=0, 
            batch_norm=False, activation=False))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


