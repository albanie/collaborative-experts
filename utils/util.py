import time
import json
import functools
import pickle
import os
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np


@functools.lru_cache(maxsize=64, typed=False)
def memcache(path, loader):
    if loader == "pkl":
        res = pickle_loader(path)
    elif loader == "npy":
        res = np_loader(path)
    else:
        raise ValueError("unknown loader: {}".format(loader))
    return res


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def pickle_loader(pkl_path):
    tic = time.time()
    print("loading features from {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    print("done in {:.3f}s".format(time.time() - tic))
    return data


def np_loader(np_path, l2norm=False):
    tic = time.time()
    print("loading features from {}".format(np_path))
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1")
    print("done in {:.3f}s".format(time.time() - tic))
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def compute_dims(config):
    face_dim = config["experts"]["face_dim"]
    arch_args = config["arch"]["args"]
    vlad_clusters = arch_args["vlad_clusters"]

    if arch_args["use_ce"]:

        expert_modality_dim = OrderedDict([
            ("face", (face_dim, 512)),
            ("audio", (128 * vlad_clusters["audio"], 512)),
            ("rgb", (2048, 512)),
            ("speech", (300 * vlad_clusters["speech"], 512)),
            ("ocr", (300 * vlad_clusters["ocr"], 512)),
            ("flow", (1024, 512)),
            ("scene", (2208, 512)),
        ])
    else:
        expert_modality_dim = OrderedDict([
            ("face", (face_dim, face_dim)),
            ("audio", (128 * vlad_clusters["audio"], 128)),
            ("rgb", (2048, 2048)),
            ("speech", (300 * vlad_clusters["speech"], 300)),
            ("ocr", (300 * vlad_clusters["ocr"], 300)),
            ("flow", (1024, 1024)),
            ("scene", (2208, 512)),
        ])

    raw_input_dims = dict()
    for modality, dim_pair in expert_modality_dim.items():
        raw_dim = dim_pair[0]
        if modality in {"audio", "speech", "ocr"}:
            raw_dim = raw_dim // vlad_clusters[modality]
        raw_input_dims[modality] = raw_dim

    if config["experts"]["drop_feats"]:
        to_drop = config["experts"]["drop_feats"].split(",")
        print("dropping: {}".format(to_drop))
        for drop in to_drop:
            del expert_modality_dim[drop]

    return expert_modality_dim, raw_input_dims


def ensure_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return x


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
