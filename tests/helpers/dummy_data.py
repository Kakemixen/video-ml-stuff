import numpy as np
import pandas as pd
import cv2
from shutil import rmtree
import torch

from utils import io_utils


TMP_BASE_DIR = "/tmp/ml_project_video_dummy_data/"

def create_dummy_data(n_frames=8*3, vid_length=8):
    io_utils.make_folder_structure(TMP_BASE_DIR)
    shapes = [ (np.random.randint(16,32),np.random.randint(16,32),3) 
            for _ in range(n_frames)]
    frames = [np.random.randint(0, 255, shapes[i])
            for i in range(n_frames)]
    segmentations = [np.random.randint(0, 40, shapes[i])
            for i in range(n_frames)]
    frame_paths = [ f"{TMP_BASE_DIR}dummy_frames/uuid_{i}.jpeg"
            for i in range(n_frames)]
    segmentation_paths = [ f"{TMP_BASE_DIR}dummy_segmentations/uuid_{i}.png"
            for i in range(n_frames)]

    for i in range(n_frames):
        io_utils.write_img(frames[i], frame_paths[i])
        io_utils.write_img(segmentations[i], segmentation_paths[i])

    indices = pd.MultiIndex.from_tuples(
            [((i//vid_length)+1, i%vid_length) 
            for i in range(n_frames)])
    df = pd.DataFrame({
                "video_frame": frame_paths, 
                "annotation_frame":segmentation_paths},
            index=indices)
    df_path = f"{TMP_BASE_DIR}samples.csv"
    df.to_csv(df_path)

    return df_path, df, {"input": frames, "segmentation":segmentations}
    

def remove_dummy_data():
    rmtree(TMP_BASE_DIR)

def create_dummy_batch(batch_size=5, channels=3, height=4, width=6):
    rand_pred = torch.rand((5,3,4,6))
    target = torch.rand((5,3,4,6))
    corr_pred = target.clone()
    return rand_pred, corr_pred, target

