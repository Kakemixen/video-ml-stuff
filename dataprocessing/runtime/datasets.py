import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import io_utils, img_utils
from dataprocessing.runtime.preprocessing import normalize, batch_to_tensor
import torch


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_size=(360, 640)):
        super().__init__()
        indices = np.unique([x[0] for x in df.index]).tolist()
        self._df = df.set_index(pd.MultiIndex.from_tuples(
            [(indices.index(x[0]), x[1]) for x in df.index]
        ))

        self._sample_length = 5
        self.img_size = img_size

    def _prepare(self, frame, segmentation):
        assert frame.shape[:2] == segmentation.shape[:2], f"frame of shape {frame.shape} does not match segmentation of shape {segmentation.shape}"
        frame = img_utils.resize_img(frame, self.img_size, interp=img_utils.interp_method.linear)
        segmentation = img_utils.resize_img(segmentation, self.img_size, interp=img_utils.interp_method.nearest)
        h1, h2, w1, w2 = img_utils.random_crop(frame, self.img_size)
        frame = frame[h1:h2, w1:w2, :]
        segmentation = segmentation[h1:h2, w1:w2, :]
        return frame, segmentation

    def __len__(self):
        return len(np.unique([x[0] for x in self._df.index]))

    def __getitem__(self, idx):
        video = self._df.loc[idx]
        assert len(video) <= self._sample_length

        start = np.random.randint(0, len(video) - self._sample_length)

        sample_range = range(start, start + self._sample_length)
        frames = (video[f"video_frame"][i] for i in sample_range)
        annotations = (video[f"annotation_frame"][i] for i in sample_range)

        sample = {"input": io_utils.read_img_batch(frames),
                  "segmentation": io_utils.read_img_batch(annotations)}

        # do zip magic
        sample["input"], sample["segmentation"] = zip(*[self._prepare(f,s) for f,s in 
            zip(sample["input"], sample["segmentation"])])

        sample["input"] = normalize(batch_to_tensor(sample["input"]))
        sample["segmentation"] = batch_to_tensor(sample["segmentation"])

        return sample
