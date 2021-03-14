import cv2
import numpy as np
import json
import os

def read_img(path):
    return cv2.imread(path)

def write_img(img, path):
    make_folder_structure(path)
    return cv2.imwrite(path, img)

def write_npy(arr, path):
    make_folder_structure(path)
    return np.save(path, arr)
    
def make_folder_structure(path):
    dir_ = os.path.dirname(path)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)


if __name__ == "__main__":
    path = "data/train/JPEGImages/0043f083b5/00015.jpg"
    json_path = "data/train/instances.json"
    with open(json_path) as f:
        data = json.load(f)
    import pdb; pdb.set_trace()
