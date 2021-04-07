import pytest
import torch
import pandas as pd

from dataprocessing.runtime.datasets import VideoDataset

from tests.helpers import dummy_data

class TestVideoDataset:

    @classmethod
    def setup_class(cls):
        _, cls.df, _ = dummy_data.create_dummy_data()

    @classmethod
    def teardown_class(cls):
        dummy_data.remove_dummy_data()
        
    def test_loading_runs(self):
        dataset = VideoDataset(self.df)
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None
            assert isinstance(sample, dict)
            assert sample["input"].dtype is torch.float32
            assert sample["input"].dim() == 4
            assert sample["segmentation"].dtype is torch.uint8
            assert sample["segmentation"].dim() == 4
            assert torch.max(sample["input"]) <= 1.0
            assert torch.max(sample["segmentation"]) <= 40
            assert sample["input"].shape[0] == sample["segmentation"].shape[0]
            assert sample["input"].shape[1] == 3
            assert sample["segmentation"].shape[1] == 1
            assert sample["input"].shape[2:] == sample["segmentation"].shape[2:]
            if i >= 5: break

    def test_indexing_with_non_range_indices(self):
        df = self.df.set_index(pd.MultiIndex.from_tuples(
            [(x[0]*4, x[1]) for x in self.df.index]
        ))
        dataset = VideoDataset(df)
        for i in range(len(dataset)):
            sample = dataset[i]
            if i >= 5: break

