import cv2
import numpy as np
from enum import Enum

class interp_method(Enum):
    nearest = 1
    linear = 2
    cubic = 3


def resize_img(img, size, keep_aspect_ratio=True, interp=interp_method.nearest):
    """
    img: numpy array from cv2 in HWC, 0-255
    Resize imp to desired size, prioritizing H dim if keepint aspect ratio
    """
    if interp == interp_method.nearest:
        cv2_interp = cv2.INTER_NEAREST
    elif interp == interp_method.linear:
        cv2_interp = cv2.INTER_LINEAR
    elif interp == interp_method.cubic:
        cv2_interp = cv2.INTER_CUBIC
    else:
        raise ValueError

    if not keep_aspect_ratio:
        return cv2.resize(img, (size[1], size[0]), 
                interpolation=cv2_interp)

    # set ratio so that it is the smallest possible, but no smaller than new_size
    ratio = max(size[0]/img.shape[0], size[1]/img.shape[1])
    new_size = (int(img.shape[0] * ratio + 0.5), int(img.shape[1] * ratio + 0.5))
    
    return cv2.resize(img, (int(new_size[1]), int(new_size[0])), \
                interpolation=cv2_interp)

    
def random_crop(img, size, h_range=[0.0, 1.0], w_range=[0.0, 1.0]):
    """
    Generate slice indices to crop images of same shape as param:img
    size is in H,W
    """
    h_relative = h_range if isinstance(h_range, (int, float)) else np.random.uniform(*h_range)
    w_relative = w_range if isinstance(w_range, (int, float)) else np.random.uniform(*w_range)

    # top left corner of index
    h = int(h_relative * (img.shape[0] - size[0] - 1))
    w = int(w_relative * (img.shape[1] - size[1] - 1))

    return h, h+size[0], w, w+size[1]

    


    
