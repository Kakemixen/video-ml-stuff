import cv2
import numpy as np
import torch 

def normalize(img):
    return img / 255.0

def batch_to_tensor(batch):
    """ 
    batch: list of numpy arrays (images from cv2 
    Stack the list to a tensor and go from HWC -> CHW
    """
    tensor = torch.stack([torch.from_numpy(arr) for arr in batch])
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

