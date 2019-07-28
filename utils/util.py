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
def memcache(path):
    suffix = Path(path).suffix
    if suffix in {".pkl", ".pickle"}:
        res = pickle_loader(path)
    elif suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix}")
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
        data = np.load(f, encoding="latin1", allow_pickle=True)
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


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class HashableOrderedDict(OrderedDict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def compute_dims(config, logger=None):
    if logger is None:
        logger = config.get_logger('utils')

    experts = config["experts"]
    # TODO(Samuel): clean up the logic since it's a little convoluted
    ordered = sorted(config["experts"]["modalities"])

    if experts["drop_feats"]:
        to_drop = experts["drop_feats"].split(",")
        logger.info(f"dropping: {to_drop}")
        ordered = [x for x in ordered if x not in to_drop]

    dims = []
    arch_args = config["arch"]["args"]
    msg = f"It is not valid to use both the `use_ce` and `mimic_ce_dims` options"
    assert not (arch_args["use_ce"] and arch_args.get("mimic_ce_dims", False)), msg
    for expert in ordered:
        if expert == "face":
            in_dim, out_dim = experts["face_dim"], experts["face_dim"]
        elif expert == "audio":
            in_dim, out_dim = 128 * config["arch"]["args"]["vlad_clusters"]["audio"], 128
        elif expert == "rgb":
            in_dim, out_dim = 2048, 2048
        elif expert == "speech":
            in_dim, out_dim = 300 * config["arch"]["args"]["vlad_clusters"]["speech"], 300
        elif expert == "ocr":
            in_dim, out_dim = 300 * config["arch"]["args"]["vlad_clusters"]["ocr"], 300
        elif expert == "flow":
            in_dim, out_dim = 1024, 1024
        elif expert == "scene":
            in_dim, out_dim = 2208, 512  # TODO(Samuel) - double check this

        # For the CE architecture, we need to project all features to a common
        # dimensionality
        if arch_args["use_ce"] or arch_args.get("mimic_ce_dims", False):
            out_dim = experts["ce_shared_dim"]

        dims.append((expert, (in_dim, out_dim)))
    expert_dims = OrderedDict(dims)

    # To remove the dependency of dataloader on the model architecture, we create a
    # second copy of the expert dimensions which accounts for the number of vlad
    # clusters
    raw_input_dims = OrderedDict()
    for expert, dim_pair in expert_dims.items():
        raw_dim = dim_pair[0]
        if expert in {"audio", "speech", "ocr"}:
            raw_dim = raw_dim // arch_args["vlad_clusters"][expert]
        raw_input_dims[expert] = raw_dim

    return expert_dims, raw_input_dims


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
