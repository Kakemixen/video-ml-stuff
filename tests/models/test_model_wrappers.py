import pytest
import torch

from collections.abc import Iterable

from tests.helpers import dummy_data
from unittest.mock import MagicMock
    
from models.wrappers.basic_wrappers import TrainingWrapper, VideoGeneratorWrapper

class TestTrainingWrapper:

    @classmethod
    def setup_class(cls):
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(batch_size=5)
        identity = lambda x: {"prediction": x}
        cls.wrapped = TrainingWrapper(identity)

    def test_training_wrapper_forward_pass_dict(self):

        forward_out = self.wrapped(self.rand_pred)
        assert isinstance(forward_out, dict)
        assert (forward_out["prediction"] == self.rand_pred).all()
        
    def test_training_wrapper_different_predict_in_different_out(self):
        assert (self.wrapped.predict(self.rand_pred) == self.rand_pred).all()

        assert (self.wrapped.predict(self.rand_input) != self.rand_pred).all()

    def test_training_wrapper_loss_backward_is_called(self):
        loss_return = torch.Tensor(0)
        loss_return.backward = MagicMock(name="backward_pass")
        loss_return.detach = MagicMock(name="detach_call")
        loss = lambda p,t: loss_return

        loss_out = self.wrapped.calculate_loss(self.rand_pred, self.target, loss)
        loss_return.backward.assert_called()
        loss_return.detach.is_called()

    def test_training_wrapper_loss_backward_is_suppressed(self):
        loss_return = torch.Tensor(0)
        loss_return.backward = MagicMock(name="backward_pass")
        loss_return.detach = MagicMock(name="detach_call")
        loss = lambda p,t: loss_return

        loss_out = self.wrapped.calculate_loss(self.rand_pred, self.target, 
                loss, propagate=False)
        loss_return.backward.not_called()
        loss_return.detach.is_called()

class TestVideoGenWrapper:

    @classmethod
    def setup_class(cls):
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(batch_size=5)
        cls.rand_input = torch.stack([cls.rand_input for _ in range(3)])
        cls.rand_pred  = torch.stack([cls.rand_pred for _ in range(3)])
        cls.corr_pred  = torch.stack([cls.corr_pred for _ in range(3)])
        cls.target     = torch.stack([cls.target for _ in range(3)])
        identity = lambda x: {"prediction": x}
        cls.wrapped = VideoGeneratorWrapper(identity)

    def test_training_wrapper_forward_pass_dict(self):

        forward_out = self.wrapped(self.rand_pred)
        assert isinstance(forward_out, Iterable)
        assert (next(forward_out)["prediction"] == self.rand_pred).all()
        
    def test_training_wrapper_different_predict_in_different_out(self):
        assert (next(self.wrapped.predict(self.rand_pred)) == \
                self.rand_pred).all()

        assert (next(self.wrapped.predict(self.rand_input)) != \
                self.rand_pred).all()

    def test_training_wrapper_loss_backward_is_called(self):
        loss_return = torch.Tensor(0)
        loss_return.backward = MagicMock(name="backward_pass")
        loss_return.detach = MagicMock(name="detach_call")
        loss = lambda p,t: loss_return

        loss_out = self.wrapped.calculate_loss(self.rand_pred, self.target, loss)
        loss_return.backward.assert_called()
        loss_return.detach.is_called()

    def test_training_wrapper_loss_backward_is_suppressed(self):
        loss_return = torch.Tensor(0)
        loss_return.backward = MagicMock(name="backward_pass")
        loss_return.detach = MagicMock(name="detach_call")
        loss = lambda p,t: loss_return

        loss_out = self.wrapped.calculate_loss(self.rand_pred, self.target, 
                loss, propagate=False)
        loss_return.backward.not_called()
        loss_return.detach.is_called()
