import os
from utils import io_utils
from pycocotools import mask as maskUtils
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

class YoutubeVISDecoder:
    def __init__(self, file_root_dir, json_path=None):
        self.file_root_dir = file_root_dir
        
        self.video_files = defaultdict(list) # video_id -> [frame_path, ...]
        self.video_sizes = defaultdict(list) # video_id -> [height, width]
        # video_id -> {timestep -> {category_id -> [rle, ...]
        self.annotations = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))  
        self.segm_files = defaultdict(list)  # video_id -> [frame_path, ...]   

        if json_path is not None:
            self.index_data(json_path)

    def index_data(self, json_path):
        print("reading json file")
        with open(json_path) as f:
            data = json.load(f)

        print("indexing videos")
        for video_dict in data["videos"]:
            self.video_files[video_dict["id"]] = video_dict["file_names"]
            self.video_sizes[video_dict["id"]] = [video_dict["height"], video_dict["width"]]
        
        self.video_files = ddict2dict(self.video_files)
        self.video_sizes = ddict2dict(self.video_sizes)

        print("indexing annotations")
        for ann_dict in data["annotations"]:
            for i, segm in enumerate(ann_dict["segmentations"]):
                self.annotations[ann_dict["video_id"]][i][ann_dict["category_id"]].append(segm)

        self.annotations = ddict2dict(self.annotations)

    def decode_annotations(self, dest_dir):
        if not os.path.isdir(dest_dir):
            print("making dest dir", dest_dir)
            os.makedirs(dest_dir)


        with mp.Pool(16) as pool:
            self.segm_files = {video_id: vid_segm_files for video_id, vid_segm_files in zip(
                (v for v in self.video_files.keys()), 
                tqdm(pool.imap(worker, 
                    ((self.annotations[video_id], self.video_sizes[video_id], self.video_files[video_id], dest_dir ) 
                        for video_id in self.video_files.keys()
                    )
                ), desc="decoding annotations", unit="video", total=len(self.video_files)))}

    def write_samples_df(self, dest_path):
        io_utils.make_folder_structure(dest_path)

def worker(inputs):
    annotations, size, video_paths, dest_dir = inputs
    h, w = size
    segm_files = []

    for i, segms in enumerate(annotations.values()):
        mask = np.zeros([h, w], dtype=np.uint8)
        for cat_id, segm in segms.items():
            for obj in segm: # aggregating here is what makes us blend instances
                if obj is None: continue
                if type(obj) == list:
                    # polygon -- a single object might consist of multiple parts
                    # we merge all parts into one mask rle code
                    rles = maskUtils.frPyObjects(obj, h, w)
                    rle = maskUtils.merge(rles)
                elif type(obj['counts']) == list:
                    # uncompressed RLE
                    rle = maskUtils.frPyObjects(obj, h, w)
                else:
                    # rle
                    rle = ann['segmentation']
                m = maskUtils.decode(rle)
                mask += m * cat_id
        # png format is considerably smaller than numpy arrays 3kb < 900kb
        mask_path = os.path.join(dest_dir, os.path.splitext(video_paths[i])[0] + "_mask.png")
        if not io_utils.write_img(mask, mask_path):
            print("could not write mask", mask_path)
        segm_files.append(mask_path)
    return segm_files
        




def working_attempt():
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

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    #working_attempt()
    decoder = YoutubeVISDecoder("data/train/JPEGImages", "data/train/instances.json")
    decoder.decode_annotations("data/train/semantic_segmentations")
    decoder.write_samples_df("data/train/samples.csv")

