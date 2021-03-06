import pytest
import torch

from tests.helpers import dummy_data
from unittest.mock import MagicMock
    
from models.segmentors.basic_segmentors import TrivialSegmentor

class TestModelSegmentors:

    @classmethod
    def setup_class(cls):
        cls.batch_size = 5
        cls.in_c = 3
        cls.out_c = 20
        cls.h = 4
        cls.w = 8
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(
                        batch_size=cls.batch_size,
                        in_channels=cls.in_c,
                        out_channels=cls.out_c,
                        height=cls.h, width=cls.w
                        )

    def test_trivial_encoder_out_shape_down_2(self):
        encoder = TrivialSegmentor(in_c=self.in_c, out_c=self.out_c, upscale_x=2)
        
        out = encoder(self.rand_input)
        assert torch.is_tensor(out)
        assert out.dim() == 4
        assert out.shape[1] == self.out_c
        assert out.shape[2] == self.h * 2**2
        assert out.shape[3] == self.w * 2**2

