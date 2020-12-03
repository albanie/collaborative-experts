""" VaTeX dataset module.
"""
import copy
from pathlib import Path
from typing import Dict, Union, List

from typeguard import typechecked
from zsvision.zs_utils import memcache, concat_features

from utils import memory_summary
from base.base_dataset import BaseDataset


class VaTeX(BaseDataset):

    @staticmethod
    @typechecked
    def dataset_paths(training_file=-1) -> Dict[str, Union[Path, str, Dict, List[str]]]:
        subset_paths = {}
        test_splits = {
            "full-val": "val_list_split1.txt",
            "full-test": "val_list_split2.txt",
        }
        for split_name, fname in test_splits.items():
            subset_paths[split_name] = {"train": "train_list.txt", "val": fname}

        feature_names = BaseDataset.common_feat_names()
        custom_paths = {
            "audio": ["aggregated_audio/vggish-raw.hickle"],
        }
        text_feat_paths = BaseDataset.common_text_feat_paths()

        text_feat_dir = Path("text-embeddings")

        text_feat_paths = {key: text_feat_dir / fname
                           for key, fname in text_feat_paths.items()}
        challenge_text_feat_paths = {
            key: Path("aggregated_text_feats") / f"{key}{fname.suffix}"
            for key, fname in text_feat_paths.items()
        }
        feature_info = {
            "custom_paths": custom_paths,
            "feature_names": feature_names,
            "subset_list_paths": subset_paths,
            "text_feat_paths": text_feat_paths,
            "challenge_text_feat_paths": challenge_text_feat_paths,
            "raw_captions_path": "raw-captions.pkl",
        }
        return feature_info

    def load_features(self):
        if self.distil_params is not None:
            self.distil_features = {}
            d_base_path = self.distil_params['base_path']

            teachers = list(map(lambda x: d_base_path + x, self.distil_params['teachers']))

            for i, f_name in enumerate(teachers):
                self.distil_features[i] = memcache(f_name)

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
        if self.challenge_mode:
            self.load_challenge_text_features()
        else:
            self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
            text_feat_path = root_feat / self.paths["text_feat_paths"][self.text_feat]
            self.text_features = memcache(text_feat_path)

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for VaTeX, since we assume "
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 10, msg
