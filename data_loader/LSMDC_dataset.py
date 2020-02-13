""" LSMDC dataset module.
"""

import copy
import time
import torch as th
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from utils.util import ensure_tensor
from torch.utils.data import Dataset
from utils.util import memcache
from base.base_dataset import BaseDataset
from utils import memory_summary
from utils.util import memcache, ensure_tensor, concat_features


class LSMDC(BaseDataset):

    @staticmethod
    def dataset_paths(split_name, text_feat):
        assert split_name in {"full-val", "full-test"}
        subset_paths = {"train": "train_list.txt"}
        if split_name == "full-val":
            subset_paths["val"] = "val_list.txt"
        else:
            subset_paths["val"] = "test_list.txt"

        feature_names = [
            "imagenet.senet154.0",
            "scene.densenet161.0",
            "i3d.i3d.0",
            "imagenet.resnext101_32x48d.0",
            "trn.moments-trn.0",
            "r2p1d.r2p1d-ig65m.0",
            "r2p1d.r2p1d-ig65m-kinetics.0",
            "moments_3d.moments-resnet3d50.0",
            "moments-static.moments-resnet50.0",
            "detection",
            "detection-sem",
        ]
        custom_paths = {
            "audio": ["aggregated_audio/vggish-raw.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-w2v.pkl"],
            "face": ["antoine/face-max-with-blank-val.pickle"],
            "flow": ["antoine/i3d-i3d-max-fps25-stride25.pickle"],
            "speech": ["aggregated_speech/speech-w2v.pickle"],
        }
        if text_feat == "openai":
            text_feat_name = "openai-feats.pkl"
        elif text_feat == "w2v":
            text_feat_name = "w2v.pkl"
        else:
            raise ValueError(f"Text features {text_feat} not supported.")
        feature_info = {
            "custom_paths": custom_paths,
            "feature_names": feature_names,
            "subset_list_paths": subset_paths,
            "text_feat_path": Path("aggregated_text_feats") / text_feat_name,
            "raw_captions_path": "raw-captions.pkl",
        }
        return feature_info

    # def configure_train_test_splits(self, split_name):
    #     print("loading training/val splits....")
    #     tic = time.time()

        # for subset, fname in subset_lists.items():
        #     subset_list_path = Path(self.root_feat) / fname
        #     with open(subset_list_path) as f:
        #         self.partition_lists[subset] = f.read().splitlines()
        # print(f"done in {time.time() - tic:.3f}s")
        # self.split_name = split_name

    def load_features(self):
        root_feat = Path(self.root_feat)
        feat_names = {key: self.visual_feat_paths(key) for key in
                      self.paths["feature_names"]}
        feat_names.update(self.paths["custom_paths"])
        features = {}
        # modern, custom = LSMDC.supported_features(split_name=self.split_name)
        # feat_names = {key: self.visual_feat_paths(key) for key in modern}
        # feat_names.update(custom)
        # features = {}
        for expert, rel_names in feat_names.items():
            if expert not in self.ordered_experts:
                continue
            feat_paths = tuple([root_feat / rel_name for rel_name in rel_names])
            if len(feat_paths) == 1:
                features[expert] = memcache(feat_paths[0])
            else:
                # support multiple forms of feature (e.g. max and avg pooling). For
                # now, we only support direct concatenation
                msg = f"{expert}: Only direct concatenation of muliple feats is possible"
                print(f"Concatenating aggregates for {expert}....")
                assert self.feat_aggregation[expert]["aggregate"] == "concat", msg
                axis = self.feat_aggregation[expert]["aggregate-axis"]
                x = concat_features.cache_info()  # pylint: disable=no-value-for-parameter
                print(f"concat cache info: {x}")
                features_ = concat_features(feat_paths, axis=axis)
                memory_summary()

                # Make separate feature copies for each split to allow in-place filtering
                features[expert] = copy.deepcopy(features_)

        self.features = features
        # if self.text_feat == "openai":
        #     text_feat_name = "openai-feats.pkl"
        # elif self.text_feat == "w2v":
        #     text_feat_name = "w2v.pkl"
        # else:
        #     raise ValueError(f"Text features {self.text_feat} not supported.")
        # text_feat_path = Path(root_feat) / "aggregated_text_feats" / text_feat_name
        self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
        self.text_features = memcache(root_feat / self.paths["text_feat_path"])
        # self.raw_captions = memcache(Path(self.root_feat) / "raw-captions.pkl")
        # self.text_features = memcache(text_feat_path)

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for LSMDC, since we assume "
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
