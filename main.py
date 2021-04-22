import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from training.model_trainer import ModelTrainer
from models.segmentors.basic_segmentors import TrivialSegmentor
from models.encoders.basic_encoders import TrivialEncoder
from models.architectures.encoder_decoder import SimpleEncoderDecoder
from models.wrappers.basic_wrappers import VideoGeneratorWrapper
from dataprocessing.runtime.datasets import VideoDataset
from losses.wrappers import DictWrapper as LossWrapper
from losses.criteria import CELoss

def main():
    df_path = "data/train/samples.csv"
    df = pd.read_csv(df_path, index_col=[0,1])
    train_df, val_df = split_df(df)
    _, tb_df = split_df(val_df, val_num=10)
        
    batch_size = 16
    train_dataloader = DataLoader(VideoDataset(train_df), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    val_dataloader = DataLoader(VideoDataset(val_df), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    tb_dataloader = DataLoader(VideoDataset(tb_df), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    downscale = 2
    print(downscale)
    model = VideoGeneratorWrapper( SimpleEncoderDecoder(
            TrivialEncoder(in_c=3, enc_c=64, out_c=128, downscale_x=downscale),
            TrivialSegmentor(in_c=128, enc_c=64, out_c=41, upscale_x=downscale)
    ))


    loss = LossWrapper({"prediction": CELoss()})

    trainer = ModelTrainer(model, train_dataloader, val_dataloader, tb_dataloader, loss)

    trainer.train(epochs=4)

def split_df(df, train_frac=0.8, val_num=None, train_num=None):
    vid_indices = pd.Series(np.unique([x[0] for x in df.index]))
    if train_num:
        train = vid_indices.sample(n=train_num)
    elif val_num:
        train = vid_indices.sample(n=len(vid_indices) - val_num)
    elif train_frac:
        train = vid_indices.sample(frac=train_frac)
    else:
        raise ValueError("need some sample arg")
    test = vid_indices.drop(train.index)
    return df.loc[train], df.loc[test]


if __name__ == "__main__":
    np.random.seed(42)
    main()

