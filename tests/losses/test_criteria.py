import pytest
import torch

from losses import criteria

class TestLosses:

    @classmethod
    def setup_class(cls):
        cls.rand_pred = torch.rand((5,3,4,6))
        cls.target = torch.rand((5,3,4,6))
        cls.corr_pred = cls.target.clone()

    def test_bce(self):
        loss = criteria.BCELoss()
        rand_out = loss(self.rand_pred, self.target)
        corr_out = loss(self.corr_pred, self.target)
        assert rand_out.shape == torch.Size([5])
        assert corr_out.shape == torch.Size([5])
        assert ((rand_out - corr_out) > 0.0).all()


