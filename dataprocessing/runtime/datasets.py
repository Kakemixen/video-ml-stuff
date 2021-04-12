import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import io_utils, img_utils
from dataprocessing.runtime.preprocessing import normalize, batch_to_tensor
import torch


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_size=(352, 640)):
        super().__init__()
        indices = np.unique([x[0] for x in df.index]).tolist()
        self._df = df.set_index(pd.MultiIndex.from_tuples(
            [(indices.index(x[0]), x[1]) for x in df.index]
        ))

        self._sample_length = 5
        self.img_size = img_size

    def _prepare(self, frame, segmentation):
        assert frame.shape[:2] == segmentation.shape[:2], \
                f"frame of shape {frame.shape} does not match segmentation of shape {segmentation.shape}"
        frame = img_utils.resize_img(frame, self.img_size, interp=img_utils.interp_method.linear)
        segmentation = img_utils.resize_img(segmentation, self.img_size, interp=img_utils.interp_method.nearest)
        segmentation = segmentation[:, :, np.newaxis]
        h1, h2, w1, w2 = img_utils.random_crop(frame, self.img_size)
        frame = frame[h1:h2, w1:w2, :]
        segmentation = segmentation[h1:h2, w1:w2, :]
        return frame, segmentation

    def __len__(self):
        return len(np.unique([x[0] for x in self._df.index]))

    def __getitem__(self, idx):
        video = self._df.loc[idx]
        if len(video) <= self._sample_length:
            start = 0
            end = len(video)
            fill = self._sample_length - len(video)
        else:
            assert len(video) > self._sample_length
            start = np.random.randint(0, len(video) - self._sample_length)
            end = start + self._sample_length
            fill = 0

        sample_range = range(start, end)
        frames = (video[f"video_frame"][i] for i in sample_range)
        annotations = (video[f"annotation_frame"][i] for i in sample_range)

        sample = {"input": io_utils.read_img_batch(frames),
                  "segmentation": io_utils.read_img_batch(annotations, grayscale=True)}

        # do zip magic
        sample["input"], sample["segmentation"] = zip(*[self._prepare(f,s) for f,s in 
            zip(sample["input"], sample["segmentation"])])

        # convert np.array -> torch.Tensor, padding if necerrary
        sample["input"] = F.pad(normalize(batch_to_tensor(sample["input"])),    pad=(0,0,0,0,0,0,0,fill))
        sample["segmentation"] = F.pad(batch_to_tensor(sample["segmentation"]), pad=(0,0,0,0,0,0,0,fill))

        return sample
