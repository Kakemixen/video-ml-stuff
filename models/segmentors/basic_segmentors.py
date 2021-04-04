import torch
from torch import nn as nn

from models.blocks.conv_blocks import ConvBNReLU
from models.encoders.basic_encoders import TrivialEncoder

class TrivialSegmentor(nn.Module):
    def __init__(self, in_c=3, enc_c=16, latent_c=64, out_c=1, downscale_x=1):
        super().__init__()
        self.encoder = TrivialEncoder(in_c=in_c, enc_c=enc_c, out_c=latent_c, downscale_x=downscale_x)

        blocks = [ ConvBNReLU(latent_c, enc_c),
                   nn.UpsamplingBilinear2d(scale_factor=2) ]
        for _ in range(1, downscale_x):
            blocks.append(ConvBNReLU(enc_c, enc_c))
            blocks.append(nn.UpsamplingBilinear2d(scale_factor=2))
        blocks.append(ConvBNReLU(enc_c, out_c, ks=1, s=1, p=0, 
            batch_norm=False, activation=False))
        self.decoder = nn.Sequential(*blocks)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.decoder(feat)
        return feat

