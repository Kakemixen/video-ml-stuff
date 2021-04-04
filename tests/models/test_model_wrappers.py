import pytest
import torch

from tests.helpers import dummy_data
from unittest.mock import MagicMock
    
from models.wrappers.basic_wrappers import TrainingWrapper

class TestModelWrappers:

    @classmethod
    def setup_class(cls):
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(batch_size=5)

    def test_training_wrapper_forward_pass_dict(self):
        identity = lambda x: {"prediction": x}
        wrapped = TrainingWrapper(identity)

        forward_out = wrapped(self.rand_pred)
        assert isinstance(forward_out, dict)
        assert (forward_out["prediction"] == self.rand_pred).all()
        
    def test_training_wrapper_different_predict_in_different_out(self):
        identity = lambda x: {"prediction": x}
        wrapped = TrainingWrapper(identity)
        assert (wrapped.predict(self.rand_pred) == self.rand_pred).all()

        assert (wrapped.predict(self.rand_input) != self.rand_pred).all()

    def test_training_wrapper_loss_backward_is_called(self):
        identity = lambda x: {"prediction": x}
        wrapped = TrainingWrapper(identity)

        loss_return = torch.Tensor(0)
        loss_return.backward = MagicMock()
        loss_return.backward._mock_name = "backward pass"
        loss = lambda p,t: loss_return
        loss_out = wrapped.calculate_loss(self.rand_pred, self.target, loss)
        loss_return.backward.assert_called()

