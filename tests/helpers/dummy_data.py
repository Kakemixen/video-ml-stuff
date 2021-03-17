import numpy as np
import pandas as pd
import cv2
from shutil import rmtree

from utils import io_utils


TMP_BASE_DIR = "/tmp/ml_project_video_dummy_data/"

def create_dummy_data(n_frames=8*3, vid_length=8):
    io_utils.make_folder_structure(TMP_BASE_DIR)
    frames = [np.random.randint(0, 255, (24,32,3)) 
            for _ in range(n_frames)]
    segmentations = [np.random.randint(0, 40, (24,32,1))
            for _ in range(n_frames)]
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
