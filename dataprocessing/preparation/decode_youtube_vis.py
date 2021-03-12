from utils import io_utils
from pycocotools import mask as maskUtils
import json
import numpy as np

if __name__ == "__main__":
    json_path = "data/train/instances.json"
    with open(json_path) as f:
        data = json.load(f)
            
    segms = data["annotations"][0]["segmentations"]
    idx = 14
    segm = segms[idx]
    # TODO get from annotations/0
    h = 720
    w = 1280


    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']

    m = maskUtils.decode(rle)
    print(m.shape)

