import pytest
import torch

from tests.helpers import dummy_data

from losses import criteria
from losses import wrappers

class TestLosses:

    @classmethod
    def setup_class(cls):
        cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(batch_size=5)
        cls.rand_pred = {"prediction": cls.rand_pred}
        cls.corr_pred = {"prediction": cls.corr_pred}

    def test_dict_wrapper_only_prediction(self):
        loss = criteria.BCELoss()
        wrapped = wrappers.DictWrapper({"prediction": loss})
        rand_out = wrapped(self.rand_pred, self.target)
        corr_out = wrapped(self.corr_pred, self.target)
        assert rand_out.shape == torch.Size([])
        assert corr_out.shape == torch.Size([])
        assert corr_out > 0.0
        assert rand_out > 0.0
        assert rand_out - corr_out


