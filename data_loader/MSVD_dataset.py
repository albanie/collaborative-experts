import copy
from pathlib import Path
from typing import Dict, Union, List
from collections import defaultdict

import numpy as np
from typeguard import typechecked
from zsvision.zs_utils import memcache, concat_features

from utils.util import memory_summary
from base.base_dataset import BaseDataset


class MSVD(BaseDataset):

    @staticmethod
    @typechecked
    def dataset_paths(training_file=None) -> Dict[str, Union[str, List[str], Path, Dict]]:
        subset_paths = {}
        test_splits = {
            "dev": "val_list.txt",
            "official": "test_list.txt",
            "public_server_val": "public_server_val.txt",
            "public_server_test": "public_server_test.txt",
        }

        for split_name, fname in test_splits.items():
            if training_file is not None:
                subset_paths[split_name] = {"train": training_file, "val": fname}
            else:
                subset_paths[split_name] = {"train": "train_list.txt", "val": fname}

        feature_names = BaseDataset.common_feat_names()
        custom_paths = {
            "face": ["aggregated_face_feats/face-avg.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-w2v.pickle"],
            "r2p1d.r2p1d-ig65m-kinetics.0": ["aggregated_r2p1d_30fps_256px_stride8_offset0_inner_stride1/r2p1d-ig65m-kinetics-avg.pickle"],
        }
        text_feat_paths = BaseDataset.common_text_feat_paths()
        text_feat_dir = Path("aggregated_text_feats")

        text_feat_paths = {key: text_feat_dir / fname
                           for key, fname in text_feat_paths.items()}
        challenge_text_feat_paths = {}
        for text_feat in ("openai",):
            text_feat_names = {key: f"{text_feat}-caption-{key}"
                               for key in {"train", "val", "test"}}
            text_feat_paths[text_feat] = {key: f"aggregated_text_feats/{val}.pkl"
                                          for key, val in text_feat_names.items()}
            challenge_text_feat_paths[text_feat] = \
                f"aggregated_text_feats/{text_feat}.pkl"

        feature_info = {
            "subset_list_paths": subset_paths,
            "feature_names": feature_names,
            "custom_paths": custom_paths,
            "text_feat_paths": text_feat_paths,
            "challenge_text_feat_paths": challenge_text_feat_paths,
            "raw_captions_path": "raw-captions.pkl",
            "dict_youtube_mapping_path": "dict_youtube_mapping.pkl"
        }
        return feature_info

    def load_features(self):
        root_feat = Path(self.root_feat)
        if self.distil_params is not None:
            self.distil_features = {}
            d_base_path = self.distil_params['base_path']

            teachers = list(map(lambda x: root_feat / Path(d_base_path + x), self.distil_params['teachers']))

            for i, f_name in enumerate(teachers):
                self.distil_features[i] = memcache(f_name)

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
        if self.challenge_mode:
            self.load_challenge_text_features()
        else:
            text_feat_paths = self.paths["text_feat_paths"][self.text_feat]
            if isinstance(text_feat_paths, dict):
                text_features = memcache(root_feat / text_feat_paths["train"])
                split_names = {"dev": "val", "official": "test"}
                text_features.update(memcache(
                    root_feat / text_feat_paths[split_names[self.split_name]]))
                key_map = memcache(root_feat / self.paths["dict_youtube_mapping_path"])
                inverse_map = {val: key for key, val in key_map.items()}
                text_features = {inverse_map[key]: val for key, val in
                                 text_features.items()}
            elif isinstance(text_feat_paths, (Path, str)):
                text_features = memcache(root_feat / text_feat_paths)
            else:
                raise TypeError(f"Unexpected type {type(text_feat_paths)}")
            self.text_features = text_features
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
