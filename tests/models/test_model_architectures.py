import torch

from tests.helpers import dummy_data
from unittest.mock import MagicMock
    
from models.architectures.encoder_decoder import SimpleEncoderDecoder

class TestModelArchitectures:

    @classmethod
    def setup_class(cls):
        cls.batch_size = 5
        cls.in_c = 5
        cls.out_c = 40 
        cls.h = 16
        cls.w = 32
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(
                        batch_size=cls.batch_size,
                        in_channels=cls.in_c,
                        out_channels=cls.out_c,
                        height=cls.h, width=cls.w
                        )

    def test_simple_enc_dec_model(self):
        encoder = lambda x: torch.cat([x,x], dim=1)
        decoder = lambda x: torch.cat([x,x,x,x], dim=1)
        model = SimpleEncoderDecoder(encoder, decoder)
        
        out_dict = model(self.rand_input)
        assert isinstance(out_dict, dict)
        out = out_dict["prediction"]
        assert torch.is_tensor(out)
        assert out.dim() == 4
        assert out.shape[1] == self.out_c
        assert out.shape[2] == self.h
        assert out.shape[3] == self.w

