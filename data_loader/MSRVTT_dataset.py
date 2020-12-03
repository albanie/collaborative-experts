import copy
from typing import Dict, List, Union
from pathlib import Path

from typeguard import typechecked
from zsvision.zs_utils import memcache, concat_features

from utils import memory_summary
from base.base_dataset import BaseDataset


class MSRVTT(BaseDataset):

    @staticmethod
    @typechecked
    def dataset_paths(training_file=None) -> Dict[str, Union[str, List[str], Path, Dict]]:
        subset_paths = {}
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
        splits = {"full-val", "full-test", "miech", "jsfusion"}
        splits.update(challenge_splits)
        for split_name in splits:
            if split_name == "miech":
                # For now, we follow Antoine's approach of using the first text caption
                # for the retreival task when evaluating on his custom split.
                train_list_path = "train_list_miech.txt"
                test_list_path = "test_list_miech.txt"
            elif split_name in "jsfusion":
                train_list_path = "train_list_jsfusion.txt"
                test_list_path = "val_list_jsfusion.txt"
                # NOTE: The JSFusion split (referred to as 1k-A in the paper) uses all
                # videos, but randomly samples a single caption per video from the test
                # set for evaluation. To reproduce this evaluation, we use the indices
                # of the test captions, and restrict to this subset during eval.
                js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
            elif split_name in {"full-val", "full-test"}:
                if training_file is None:
                    train_list_path = "train_list_full.txt"
                else:
                    train_list_path = training_file
                if split_name == "full-val":
                    test_list_path = "val_list_full.txt"
                else:
                    test_list_path = "test_list_full.txt"
            elif split_name in challenge_splits:
                train_list_path = "train_list.txt"
                if split_name == "val":
                    test_list_path = f"{split_name}_list.txt"
                else:
                    test_list_path = f"{split_name}.txt"
            else:
                msg = "unrecognised MSRVTT split: {}"
                raise ValueError(msg.format(split_name))
            subset_paths[split_name] = {"train": train_list_path, "val": test_list_path}
        feature_names = BaseDataset.common_feat_names()
        custom_paths = {
            "audio": ["aggregated_audio_feats/Audio_MSRVTT_new.pickle"],
            "face": ["aggregated_face_feats/facefeats-avg.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-raw.pickle"],
            "speech": ["aggregated_speech/speech-w2v.pickle"],

        }
        custom_miech_paths = custom_paths.copy()
        custom_miech_paths.update({
            "antoine-rgb": ["antoine/resnet_features.pickle"],
            "audio": ["antoine/audio_features.pickle"],
            "flow": ["antoine/flow_features.pickle"],
            "face": ["antoine/facefeats-clone.pickle"],
        })

        text_feat_paths = BaseDataset.common_text_feat_paths()
        # include non-standard text features
        text_feat_paths["openai"] = "w2v_MSRVTT_openAIGPT.pickle"
        text_feat_dir = Path("aggregated_text_feats")

        text_feat_paths = {key: text_feat_dir / fname
                           for key, fname in text_feat_paths.items()}

        challenge_text_feat_paths = {key: f"aggregated_text_feats/{key}.pickle"
                                     for key in text_feat_paths}

        feature_info = {
            "custom_paths": custom_paths,
            "custom_miech_paths": custom_miech_paths,
            "feature_names": feature_names,
            "subset_list_paths": subset_paths,
            "text_feat_paths": text_feat_paths,
            "challenge_text_feat_paths": challenge_text_feat_paths,
            "raw_captions_path": "raw-captions.pkl",
            "js_test_cap_idx_path": js_test_cap_idx_path,
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

        if self.split_name == "miech":
            custom_path_key = "custom_miech_paths"
        else:
            custom_path_key = "custom_paths"
        feat_names.update(self.paths[custom_path_key])
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
                is_concat = self.feat_aggregation[expert]["aggregate"] == "concat"
                self.log_assert(is_concat, msg=msg)
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
            if self.split_name == "jsfusion":
                self.restrict_test_captions = memcache(
                    root_feat / self.paths["js_test_cap_idx_path"])
            self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
            text_feat_path = root_feat / self.paths["text_feat_paths"][self.text_feat]
            self.text_features = memcache(text_feat_path)

            if self.restrict_train_captions:
                # hash the video names to avoid O(n) lookups in long lists
                train_list = set(self.partition_lists["train"])
                for key, val in self.text_features.items():
                    if key not in train_list:
                        continue

                    if not self.split_name == "full-test":
                        # Note that we do not perform this sanity check for the full-test
                        # split, because the text features in the cached dataset will
                        # already have been cropped to the specified
                        # `resstrict_train_captions`
                        expect = {19, 20}
                        msg = f"expected train text feats as lists with length {expect}"
                        has_expected_feats = isinstance(val, list) and len(val) in expect
                        self.log_assert(has_expected_feats, msg=msg)

                    # restrict to the first N captions (deterministic)
                    self.text_features[key] = val[:self.restrict_train_captions]
        self.summary_stats()

    def sanity_checks(self):

        if self.num_test_captions == 20:
            if len(self.partition_lists["val"]) == 2990:
                missing = 6
            elif len(self.partition_lists["val"]) == 1000:
                missing = 2
            elif len(self.partition_lists["val"]) == 497:
                missing = 0
            else:
                raise ValueError("unrecognised test set")
            found_missing = self.query_masks.size - self.query_masks.sum()
            msg = f"Expected {missing} missing queries in MSRVTT, found {found_missing}"
            correct_missing = found_missing == missing
            self.log_assert(correct_missing, msg=msg)
