import pytest

from dataprocessing.runtime.dataloader import VideoDataset

from tests.helpers import dummy_data

class TestVideoDataset:

    @classmethod
    def setup_class(cls):
        cls.df_path, _, _ = dummy_data.create_dummy_data()

    @classmethod
    def teardown_class(cls):
        dummy_data.remove_dummy_data()
        
    def test_loading_runs(self):
        dataset = VideoDataset(self.df_path)
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None
            if i >= 5: break


