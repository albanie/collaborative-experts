import time
import copy
import numpy as np
from pathlib import Path
from utils.util import memcache
from os.path import join as pjoin
from base.base_dataset import BaseDataset
from utils import memory_summary
from utils.util import memcache, ensure_tensor, concat_features
from utils.datastructures import ExpertStore


class MSRVTT(BaseDataset):

    @staticmethod
    def dataset_paths(split_name, text_feat):
        js_test_cap_idx_path = None
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
            train_list_path = "train_list_full.txt"
            if split_name == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(split_name))
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
        ]
        if split_name == "miech":
            custom_paths = {
                "antoine-rgb": ["antoine/resnet_features.pickle"],
                "audio": ["antoine/audio_features.pickle"],
                "flow": ["antoine/flow_features.pickle"],
                "face": ["antoine/facefeats-clone.pickle"],
            }
        else:
            custom_paths = {
                "audio": ["aggregated_audio_feats/Audio_MSRVTT_new.pickle"],
                "face": ["aggregated_face_feats/facefeats-avg.pickle"],
            }
        custom_paths.update({
            "ocr": ["aggregated_ocr_feats/ocr-raw.pickle"],
            "speech": ["aggregated_speech/speech-w2v.pickle"]
        })
        text_feat_name = {
            "w2v": "w2v_MSRVTT.pickle",
            "openai": "w2v_MSRVTT_openAIGPT.pickle",
            "bertxl": "w2v_MSRVTT_transformer.pickle",
        }[text_feat]
        feature_info = {
            "custom_paths": custom_paths,
            "feature_names": feature_names,
            "subset_list_paths": subset_paths,
            "text_feat_path": Path("aggregated_text_feats") / text_feat_name,
            "raw_captions_path": "raw-captions.pkl",
            "js_test_cap_idx_path": js_test_cap_idx_path,
        }
        return feature_info


    # def configure_train_test_splits(self, split_name):
    #     print("loading training/val splits....")
    #     tic = time.time()
    #     for subset, path in paths.items():
    #         subset_list_path = Path(self.root_feat) / path
    #         with open(subset_list_path) as f:
    #             self.partition_lists[subset] = f.read().splitlines()
    #     print("done in {:.3f}s".format(time.time() - tic))
    #     self.split_name = split_name

    def load_features(self):
        root_feat = Path(self.root_feat)
        feat_names = {key: self.visual_feat_paths(key) for key in
                      self.paths["feature_names"]}
        feat_names.update(self.paths["custom_paths"])
        features = {}
        # # modern, custom = MSRVTT.supported_features(split_name=self.split_name)
        # # feat_names = {key: self.visual_feat_paths(key) for key in modern}
        # # feat_names.update(custom)
        # features = {}
        for expert, rel_names in feat_names.items():
            if expert not in self.ordered_experts:
                continue
            feat_paths = tuple([root_feat / rel_name for rel_name in rel_names])
            # if expert == "speech":
            #     # fix old-style speech
            #     features_ = memcache(feat_paths[0])
            #     for key, val in features_.items():
            #         if (not hasattr(val, "size")) or val.size == 0:
            #             features_[key] = np.nan
            #     features[expert] = copy.deepcopy(features_)
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

        # text_feat_name = {
        #     "w2v": "w2v_MSRVTT.pickle",
        #     "openai": "w2v_MSRVTT_openAIGPT.pickle",
        #     "bertxl": "w2v_MSRVTT_transformer.pickle",
        # }[self.text_feat]
        # text_feat_path = Path(self.root_feat) / f"aggregated_text_feats/{text_feat_name}"
        self.features = features
        if self.split_name == "jsfusion":
            self.restrict_test_captions = memcache(
                root_feat / self.paths["js_test_cap_idx_path"])

        # self.raw_captions = memcache(Path(self.root_feat) / "raw-captions.pkl")
        # self.raw_captions = memcache(self.paths["raw_captions_path"])
        # self.text_features = memcache(self.paths["text_feat_path"])
        self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
        self.text_features = memcache(root_feat / self.paths["text_feat_path"])

        if self.restrict_train_captions:
            # hash the video names to avoid O(n) lookups in long lists
            train_list = set(self.partition_lists["train"])
            for key, val in self.text_features.items():
                if key not in train_list:
                    continue

                if not self.split_name == "full-test":
                    # Note that we do not perform this sanity check for the full-test
                    # split, because the text features in the cached dataset will already
                    # have been cropped to the specified `resstrict_train_captions`
                    msg = "expected train text features to be lists with length 19 or 20"
                    has_expected_feats = isinstance(val, list) and len(val) in {19, 20}
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
            msg = "Expected to find two missing queries in MSRVTT for full eval"
            correct_missing = self.query_masks.sum() == self.query_masks.size - missing
            self.log_assert(correct_missing, msg=msg)
