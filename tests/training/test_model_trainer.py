import pytest
import torch

from tests.helpers import dummy_data
from unittest.mock import Mock

from models.wrappers.basic_wrappers import TrainingWrapper
from losses.wrappers import DictWrapper as LossWrapper

from training.model_trainer import ModelTrainer

class TestModelTrainer:

    @classmethod
    def setup_class(cls):
        cls.batch_size = 1
        cls.in_c = 3
        cls.out_c = 2
        cls.h = 4
        cls.w = 8
        cls.rand_input, cls.rand_pred, cls.corr_pred, cls.target = \
                dummy_data.create_dummy_batch(
                        batch_size=cls.batch_size,
                        in_channels=cls.in_c,
                        out_channels=cls.out_c,
                        height=cls.h, width=cls.w
                        )
        cls.batches = [{"input": cls.rand_input, "segmentation": cls.target} for _ in range (3)]

    def test_train_epoch(self):
        mock_loss = Mock(spec=LossWrapper, name="loss")
        mock_training_wrapper = Mock(spec=TrainingWrapper, name="training_wrapper")
        mock_training_wrapper.calculate_loss = Mock(return_value=2)
        model_trainer = ModelTrainer( mock_training_wrapper,
                self.batches, self.batches,  mock_loss)

        model_trainer.train_epoch()
        mock_training_wrapper.calculate_loss.assert_called_with(
                {"input":self.rand_input, "segmentation":self.target}, mock_loss, propagate=True)


        
    def test_validate(self):
        mock_loss = Mock(spec=LossWrapper, name="loss")
        mock_training_wrapper = Mock(spec=TrainingWrapper, name="training_wrapper")
        mock_training_wrapper.calculate_loss = Mock(return_value=2)
        model_trainer = ModelTrainer( mock_training_wrapper,
                self.batches, self.batches,  mock_loss)

        val_loss = model_trainer.validate_epoch()
        mock_training_wrapper.calculate_loss.assert_called_with(
                {"input":self.rand_input, "segmentation":self.target}, mock_loss, propagate=False)
        assert val_loss >= 0

