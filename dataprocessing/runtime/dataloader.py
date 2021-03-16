import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import io_utils, img_utils
from dataprocessing.runtime.preprocessing import normalize, batch_to_tensor


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, df_path):
        super().__init__()
        self._df = pd.read_csv(df_path, index_col=[0,1])

        self._sample_length = 5

    def __len__(self):
        return len(self._df.index.levels[0])

    def __getitem__(self, idx):
        video = self._df.loc[idx+1]
        start = np.random.randint(0, len(video) - self._sample_length)

        sample_range = range(start, start + self._sample_length)
        frames = (video[f"video_frame"][i] for i in sample_range)
        annotations = (video[f"annotation_frame"][i] for i in sample_range)

        sample = {"input": io_utils.read_img_batch(frames),
                  "segmentation": io_utils.read_img_batch(annotations)}

        new_size = 360, # dont need second dim
        sample["input"] = [img_utils.resize_img(f, new_size) for f in sample["input"]]
        sample["segmentation"] = [img_utils.resize_img(f, new_size, 
                        interp=img_utils.interp_method.nearest) 
                for f in sample["segmentation"]]

        sample["input"] = normalize(batch_to_tensor(sample["input"]))
        sample["segmentation"] = batch_to_tensor(sample["segmentation"])

        return sample


if __name__ == "__main__":
    dataset = VideoDataset("data/train/samples.csv")
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        sample = dataset[i]
        print(sample["input"].shape)
        if i >= 500: break

