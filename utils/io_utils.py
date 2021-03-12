import cv2
import numpy
import json

def read_img(path):
    return cv2.imread(path)


if __name__ == "__main__":
    path = "data/train/JPEGImages/0043f083b5/00015.jpg"
    json_path = "data/train/instances.json"
    with open(json_path) as f:
        data = json.load(f)
    import pdb; pdb.set_trace()
