import pytest
import torch
import numpy as np

from tests.helpers import dummy_data

from main import split_df

class TestVideoDataset:

    @classmethod
    def setup_class(cls):
        _, cls.df, _ = dummy_data.create_dummy_data()

    @classmethod
    def teardown_class(cls):
        dummy_data.remove_dummy_data()

    def test_split_df(self):
        train, test = split_df(self.df, train_frac=0.8)
        train_vids = np.unique([x[0] for x in train.index])
        test_vids = np.unique([x[0] for x in test.index])
        assert np.floor(len(self.df.index.levels[0]) * 0.8) == \
                len(train_vids)
        assert np.ceil(len(self.df.index.levels[0]) * 0.2) == \
                len(test_vids)
        
        df_vids = np.unique([x[0] for x in self.df.index])
        assert np.concatenate([train_vids, test_vids]).sort() == \
                df_vids.sort()
