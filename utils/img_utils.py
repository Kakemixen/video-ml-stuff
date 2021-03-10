import cv2
import numpy

def read_img(path):
    return cv2.imread(path)

if __name__:
    path = "data/train/JPEGImages/0043f083b5/00015.jpg"
    print(read_img(path).shape)
