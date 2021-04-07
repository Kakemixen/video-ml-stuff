import pytest
import torch
import pandas as pd
import numpy as np

from tests.helpers import dummy_data

from utils import io_utils

class TestVideoDataset:

    @classmethod
    def setup_class(cls):
        _, cls.df, _ = dummy_data.create_dummy_data()

    @classmethod
    def teardown_class(cls):
        dummy_data.remove_dummy_data()
        
    def test_read_img(self):
        for i in range(len(self.df)):
            img = io_utils.read_img(self.df["video_frame"].iloc[i])
            mask = io_utils.read_img(self.df["annotation_frame"].iloc[i], 
                    grayscale=True)
            assert img is not None
            assert len(img.shape) == 3
            assert len(mask.shape) == 3
            assert img.shape[2] == 3
            assert mask.shape[2] == 1
            assert img.shape[:2] == mask.shape[:2]
            if i >= 5: break
