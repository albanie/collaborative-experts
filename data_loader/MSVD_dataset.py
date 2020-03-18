import copy
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
from zsvision.zs_beartype import beartype
from zsvision.zs_utils import memcache

from utils.util import memory_summary, concat_features
from base.base_dataset import BaseDataset


class MSVD(BaseDataset):

    @staticmethod
    @beartype
    def dataset_paths(split_name: str, text_feat: str) -> Dict:
        train_list_path = "train_list.txt"
        if split_name == "official":
            test_list_path = "test_list.txt"
        elif split_name == "dev":
            test_list_path = "val_list.txt"
        else:
            raise ValueError(f"unrecognised MSVD split: {split_name}")
        subset_paths = {"train": train_list_path, "val": test_list_path}
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
            "detection-sem"
        ]
        custom_paths = {
            "face": ["aggregated_face_feats/face-avg.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-w2v.pickle"],
        }
        text_feat_names = {key: f"{text_feat}-caption-{key}"
                           for key in {"train", "val", "test"}}
        text_feat_paths = {key: f"aggregated_text_feats/{val}.pkl"
                           for key, val in text_feat_names.items()}
        feature_info = {
            "subset_list_paths": subset_paths,
            "feature_names": feature_names,
            "custom_paths": custom_paths,
            "text_feat_paths": text_feat_paths,
            "raw_captions_path": "raw-captions.pkl",
            "dict_youtube_mapping_path": "dict_youtube_mapping.pkl"
        }
        return feature_info

    def load_features(self):
        root_feat = Path(self.root_feat)
        feat_names = {key: self.visual_feat_paths(key) for key in
                      self.paths["feature_names"]}
        feat_names.update(self.paths["custom_paths"])
        features = {}
        for expert, rel_names in feat_names.items():
            if expert not in self.ordered_experts:
                continue
            feat_paths = tuple([root_feat / rel_name for rel_name in rel_names])
            if len(feat_paths) == 1:
                features[expert] = memcache(feat_paths[0])
            else:
                # support multiple forms of feature (e.g. max and avg pooling). For
                # now, we only support direct concatenation
                msg = f"{expert}: Only direct concat of muliple feats is possible"
                print(f"Concatenating aggregates for {expert}....")
                assert self.feat_aggregation[expert]["aggregate"] == "concat", msg
                axis = self.feat_aggregation[expert]["aggregate-axis"]
                x = concat_features.cache_info()  # pylint: disable=no-value-for-parameter
                print(f"concat cache info: {x}")
                features_ = concat_features(feat_paths, axis=axis)
                memory_summary()

                if expert == "speech":
                    features_defaults = defaultdict(lambda: np.zeros((1, 300)))
                    features_defaults.update(features_)
                    features_ = features_defaults

                # Make separate feature copies for each split to allow in-place filtering
                features[expert] = copy.deepcopy(features_)

        self.features = features
        text_feat_paths = self.paths["text_feat_paths"]
        text_features = memcache(root_feat / text_feat_paths["train"])
        split_names = {"dev": "val", "official": "test"}
        text_features.update(memcache(
            root_feat / text_feat_paths[split_names[self.split_name]]))
        key_map = memcache(root_feat / self.paths["dict_youtube_mapping_path"])
        inverse_map = {}
        for key, value in key_map.items():
            inverse_map[value] = key
        self.text_features = {inverse_map[key]: val for key, val in text_features.items()}
        self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])

        if "detection" in self.ordered_experts:
            # Example processing
            processed = {}
            for key, subdict in self.features["detection"].items():
                box, conf = subdict["detection_boxes"], subdict["detection_scores"]
                raw = subdict["raw_feats_avg"]
                processed[key] = np.concatenate((box, conf.reshape(-1, 1), raw), axis=1)
            self.features["detection"] = processed

        if "openpose" in self.ordered_experts:
            # Example processing
            processed = {}
            for key, subdict in self.features["openpose"].items():
                raw = np.concatenate(subdict["matrix"], axis=1)
                processed[key] = raw.transpose(1, 0, 2).reshape(-1, 3 * 18)
            self.features["openpose"] = processed

    def sanity_checks(self):
        assert self.num_test_captions == 81, "Expected to have 81 test caps for MSVD"
