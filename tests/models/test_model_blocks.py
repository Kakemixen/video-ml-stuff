import pytest
import torch

from tests.helpers import dummy_data
from unittest.mock import MagicMock
    
from models.blocks import conv_blocks

class TestModelBlocks:

    @classmethod
    def setup_class(cls):
        cls.batch_size = 5
        cls.in_c = 8
        cls.out_c = 16
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(
                        batch_size=cls.batch_size,
                        in_channels=cls.in_c,
                        out_channels=cls.out_c
                        )

    def test_conv_bn_relu(self):
        block = conv_blocks.ConvBNReLU(in_c=self.in_c, out_c=self.out_c)

        forward_out = block(self.rand_input)
        assert torch.is_tensor(forward_out)
        assert forward_out.shape[1] == self.out_c
        assert forward_out.shape[2] == 4
        assert forward_out.shape[3] == 6

    def test_conv_bn_relu_stride_2(self):
        block = conv_blocks.ConvBNReLU(in_c=self.in_c, out_c=self.out_c, s=2)

        forward_out = block(self.rand_input)
        assert torch.is_tensor(forward_out)
        assert forward_out.shape[1] == self.out_c
        assert forward_out.shape[2] == 2
        assert forward_out.shape[3] == 3

    def test_conv_bn_relu_kwargs_present(self):
        block = conv_blocks.ConvBNReLU(in_c=self.in_c, out_c=self.out_c,
                ks=5, s=1, p=2, d=1, g=2, b=False,
                batch_norm=True, activation=True)

