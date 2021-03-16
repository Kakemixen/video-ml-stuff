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

    ratio = size[0] / img.shape[0]
    new_size = (img.shape[0] * ratio, img.shape[1] * ratio)
    for s in new_size:
        assert np.floor(s) == s
    
    return cv2.resize(img, (int(new_size[1]), int(new_size[0])), \
                interpolation=cv2_interp)

    
    
