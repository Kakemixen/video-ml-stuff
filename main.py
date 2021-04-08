import pandas as pd
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
        
    train_dataloader = DataLoader(VideoDataset(train_df), batch_size=32, shuffle=True, pin_memory=True, num_workers=10)
    val_dataloader = DataLoader(VideoDataset(val_df), batch_size=32, shuffle=True, pin_memory=True, num_workers=10)

    model = VideoGeneratorWrapper( SimpleEncoderDecoder(
            TrivialEncoder(in_c=3, enc_c=32, out_c=128, downscale_x=4),
            TrivialSegmentor(in_c=128, enc_c=32, out_c=41, upscale_x=4)
    ))


    loss = LossWrapper({"prediction": CELoss()})

    trainer = ModelTrainer(model, train_dataloader, val_dataloader, loss)

    trainer.train(epochs=5)


def split_df(df, train_frac=0.8):
    vid_indices = pd.Series(df.index.levels[0])
    train = vid_indices.sample(frac=train_frac)
    test = vid_indices.drop(train.index)
    return df.loc[train], df.loc[test]


if __name__ == "__main__":
    main()

