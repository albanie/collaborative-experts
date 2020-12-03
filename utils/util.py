"""
Exclude from autoreload
%aimport -util.utils
"""
import os
import copy
import json
import uuid
import random
import logging
import itertools
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import psutil
import humanize
from PIL import Image
from typeguard import typechecked


@typechecked
def filter_cmd_args(cmd_args: List[str], remove: List[str]) -> List[str]:
    drop = []
    for key in remove:
        if key not in cmd_args:
            continue
        pos = cmd_args.index(key)
        drop.append(pos)
        if len(cmd_args) > (pos + 1) and not cmd_args[pos + 1].startswith("--"):
            drop.append(pos + 1)
    for pos in sorted(drop, reverse=True):
        cmd_args.pop(pos)
    return cmd_args


@typechecked
def get_short_uuid() -> str:
    """Return a 7 alpha-numeric character random string.  We could use the full uuid()
    for better uniqueness properties, but it makes the filenames long and its not
    needed for our purpose (simply grouping experiments that were run with the same
    configuration).
    """
    return str(uuid.uuid4()).split("-")[0]


@typechecked
def parse_grid(x: str) -> Dict[str, List[str]]:
    """Parse compact command line strings of the form:
        --key1 val_a|val_b --key2 val_c|val_d

    (here a vertical bar represents multiple values)

    into a grid of separate strings e.g:
        --key1 val_a --key2 val_c
        --key1 val_a --key2 val_d
        --key1 val_b --key2 val_c
        --key1 val_b --key2 val_d

    """
    args = x.split(" ")
    group_id = get_short_uuid()
    grid_opts, parsed = {}, []
    for ii, token in enumerate(args):
        if "|" in token:
            grid_opts[ii] = token.split("|")
    grid_idx, grid_vals = [], []
    for ii, val in grid_opts.items():
        grid_idx.append(ii)
        grid_vals.append(val)
    grid_vals = list(itertools.product(*grid_vals))
    for cfg in grid_vals:
        base = copy.deepcopy(args)
        for ii, val in zip(grid_idx, cfg):
            base[ii] = val
        base.append(f"--group_id {group_id}")
        parsed.append(" ".join(base))
    return {group_id: parsed}


@typechecked
def set_seeds(seed: int):
    """Set seeds for randomisation libraries.

    Args:
        seed: the seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def update_src_web_video_dir(config):
    """Provide backwards compatible support for web directories

    Args:
        config: a configuration object containing experiment paths
    """
    src_video_dir = Path(config["visualizer"]["args"]["src_video_dir"])
    dataset = config["data_loader"]["args"]["dataset_name"]
    if dataset not in str(src_video_dir):
        lookup = {
            "ActivityNet": "activity-net/videos",
            "MSRVTT": "MSRVTT/videos/all",
            "MSVD": "MSVD/videos",
            "DiDeMo": "DiDeMo/videos",
            "LSMDC": "LSMDC/videos",
            "YouCook2": "YouCook2/videos",
            "QuerYD": "QuerYD/videos",
            "QuerYDSegments": "QuerYDSegments/videos",
        }
        src_video_dir = Path(src_video_dir.parts[0]) / lookup[dataset]
    config["visualizer"]["args"]["src_video_dir"] = Path(src_video_dir)


def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)


def flatten_dict(x, keysep="-"):
    flat_dict = {}
    for key, val in x.items():
        if isinstance(val, dict):
            flat_subdict = flatten_dict(val)
            flat_dict.update({f"{key}{keysep}{subkey}": subval
                              for subkey, subval in flat_subdict.items()})
        else:
            flat_dict.update({key: val})
    return flat_dict


def expert_tensor_storage(experts, feat_aggregation):
    expert_storage = {"fixed": set(), "variable": set(), "flaky": set()}
    # fixed_sz_experts, variable_sz_experts, flaky_experts = set(), set(), set()
    for expert, config in feat_aggregation.items():
        if config["temporal"] in {"vlad"}:
            expert_storage["variable"].add(expert)
        elif all([x in {"avg", "max", "ent", "std"} for x in config["temporal"].split("-")]):
            expert_storage["fixed"].add(expert)
        else:
            raise ValueError(f"unknown temporal strategy: {config['temporal']}")
        # some "flaky" experts are only available for a fraction of videos - we need
        # to pass this information (in the form of indices) into the network for any
        # experts present in the current dataset
        if config.get("flaky", False):
            expert_storage["flaky"].add(expert)

    # we only allocate storage for experts used by the current dataset
    for key, value in expert_storage.items():
        expert_storage[key] = value.intersection(set(experts))
    return expert_storage


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def path2str(x):
    """Recursively convert pathlib objects to strings to enable serialization"""
    for key, val in x.items():
        if isinstance(val, dict):
            path2str(val)
        elif isinstance(val, Path):
            x[key] = str(val)


def write_json(content, fname, paths2strs=False):
    if paths2strs:
        path2str(content)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in itertools.repeat(data_loader):
        yield from loader


def compute_trn_config(config, logger=None):
    trn_config = {}
    feat_agg = config["data_loader"]["args"]["feat_aggregation"]
    for static_expert in feat_agg.keys():
        if static_expert in feat_agg:
            if "trn_seg" in feat_agg[static_expert].keys():
                trn_config[static_expert] = feat_agg[static_expert]["trn_seg"]
    return trn_config


@typechecked
def compute_dims(
        config,
        logger: logging.Logger = None,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int], int]:
    if logger is None:
        logger = config.get_logger('utils')

    experts = config["experts"]
    # TODO(Samuel): clean up the logic since it's a little convoluted
    ordered = sorted(config["experts"]["modalities"])

    if experts["drop_feats"]:
        to_drop = experts["drop_feats"].split(",")
        logger.info(f"dropping: {to_drop}")
        ordered = [x for x in ordered if x not in to_drop]

    feat_agg = config["data_loader"]["args"]["feat_aggregation"]
    dims = []
    arch_args = config["arch"]["args"]
    vlad_clusters = arch_args["vlad_clusters"]
    msg = f"It is not valid to use both the `use_ce` and `mimic_ce_dims` options"
    assert not (arch_args["use_ce"] and arch_args.get("mimic_ce_dims", False)), msg
    for expert in ordered:
        temporal = feat_agg[expert]["temporal"]
        if expert == "face":
            in_dim, out_dim = experts["face_dim"], experts["face_dim"]
        elif expert in {"audio", "audio.vggish.0"} and temporal == "vlad":
            in_dim, out_dim = 128 * vlad_clusters["audio"], 128
        elif expert == "speech" and temporal == "vlad":
            in_dim, out_dim = 300 * vlad_clusters["speech"], 300
        elif expert == "ocr" and temporal == "vlad":
            in_dim, out_dim = 300 * vlad_clusters["ocr"], 300
        elif expert == "detection":
            # allow for avg pooling
            det_clusters = arch_args["vlad_clusters"].get("detection", 1)
            in_dim, out_dim = 1541 * det_clusters, 1541
        elif expert == "detection-sem":
            if config["data_loader"]["args"].get("spatial_feats", False):
                base = 300 + 16
            else:
                base = 300 + 5
            det_clusters = arch_args["vlad_clusters"].get("detection-sem", 1)
            in_dim, out_dim = base * det_clusters, base
        elif expert == "openpose":
            base = 54
            det_clusters = arch_args["vlad_clusters"].get("openpose", 1)
            in_dim, out_dim = base * det_clusters, base
        else:
            common_dim = feat_agg[expert]["feat_dims"][feat_agg[expert]["type"]]
            # account for aggregation of multilpe forms (e.g. avg + max pooling)
            common_dim = common_dim * len(feat_agg[expert]["temporal"].split("-"))
            in_dim, out_dim = common_dim, common_dim

        # For the CE architecture, we need to project all features to a common
        # dimensionality
        is_ce = config["arch"]["type"] == "CENet"
        if is_ce and (arch_args["use_ce"] or arch_args.get("mimic_ce_dims", False)):
            out_dim = experts["ce_shared_dim"]

        dims.append((expert, (in_dim, out_dim)))
    expert_dims = OrderedDict(dims)

    if vlad_clusters["text"] == 0:
        msg = "vlad can only be disabled for text with single tokens"
        assert config["data_loader"]["args"]["max_tokens"]["text"] == 1, msg

    if config["experts"]["text_agg"] == "avg":
        if hasattr(config["arch"]["args"], "vlad_clusters"):
            msg = "averaging can only be performed with text using single tokens"
            assert config["arch"]["args"]["vlad_clusters"]["text"] == 0, msg
        assert config["data_loader"]["args"]["max_tokens"]["text"] == 1

    # To remove the dependency of dataloader on the model architecture, we create a
    # second copy of the expert dimensions which accounts for the number of vlad
    # clusters
    raw_input_dims = OrderedDict()
    for expert, dim_pair in expert_dims.items():
        raw_dim = dim_pair[0]
        if expert in {"audio", "speech", "ocr", "detection", "detection-sem", "openpose",
                      "speech.mozilla.0"}:
            if feat_agg[expert]["temporal"] == "vlad":
                raw_dim = raw_dim // vlad_clusters.get(expert, 1)
        raw_input_dims[expert] = raw_dim

    with open(config["text_embedding_model_configs"], "r") as f:
        text_embedding_model_configs = json.load(f)
    text_dim = text_embedding_model_configs[experts["text_feat"]]["dim"]

    return expert_dims, raw_input_dims, text_dim


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
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
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
