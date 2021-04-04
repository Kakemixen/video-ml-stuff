import pytest
import torch

from tests.helpers import dummy_data

from losses import criteria

class TestLosses:

    @classmethod
    def setup_class(cls):
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(batch_size=5)

    def test_bce(self):
        loss = criteria.BCELoss()
        rand_out = loss(self.rand_pred, self.target)
        corr_out = loss(self.corr_pred, self.target)
        assert rand_out.shape == torch.Size([5])
        assert corr_out.shape == torch.Size([5])
        assert (corr_out > 0.0).all()
        assert (rand_out > 0.0).all()
        assert ((rand_out - corr_out) > 0.0).all()


