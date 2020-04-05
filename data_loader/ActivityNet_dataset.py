import copy
from typing import Dict, Union, List
from pathlib import Path

from zsvision.zs_utils import memcache, concat_features
from typeguard import typechecked

from utils import memory_summary
from base.base_dataset import BaseDataset


class ActivityNet(BaseDataset):

    @staticmethod
    @typechecked
    def dataset_paths() -> Dict[str, Union[str, List[str], Path, Dict]]:
        subset_paths = {}
        test_splits = {
            "val1": "val_1_list.txt",
            "val": "val_list.txt",
            "public_server_val": "public_server_val.txt",
            "public_server_test": "public_server_test.txt",
        }
        for split_name, fname in test_splits.items():
            subset_paths[split_name] = {"train": "train_list.txt", "val": fname}

        feature_names = [
            "imagenet.senet154.0",
            "scene.densenet161.0",
            "i3d.i3d.0",
            "s3dg.s3dg.0",
            "imagenet.resnext101_32x48d.0",
            "trn.moments-trn.0",
            "r2p1d.r2p1d-ig65m.0",
            "r2p1d.r2p1d-ig65m-kinetics.0",
            "moments_3d.moments-resnet3d50.0",
            "moments-static.moments-resnet50.0",
        ]
        custom_paths = {
            "audio": ["aggregated_audio/vggish-audio-raw.pickle"],
            "speech": ["aggregated_speech/goog_w2v-speech-raw.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-w2v.pkl"],
            "face": ["aggregated_facefeats_25fps_256px_stride1/face-avg.pickle"],
        }
        text_feat_paths = {}
        challenge_text_feat_paths = {}
        for text_feat in ("openai", ):
            text_feat_names = {key: f"{text_feat}-{key}"
                               for key in {"train", "val1"}}
            text_feat_paths[text_feat] = {key: f"aggregated_text_feats/{val}.pkl"
                                          for key, val in text_feat_names.items()}
            challenge_text_feat_paths[text_feat] = \
                f"aggregated_text_feats/{text_feat}.pkl"
        feature_info = {
            "custom_paths": custom_paths,
            "feature_names": feature_names,
            "subset_list_paths": subset_paths,
            "text_feat_paths": text_feat_paths,
            "challenge_text_feat_paths": challenge_text_feat_paths,
            "raw_captions_path": "raw-captions-train-val_1.pkl",
        }
        return feature_info

    def load_features(self):
        root_feat = self.root_feat
        feat_names = {key: self.visual_feat_paths(key) for key in
                      self.paths["feature_names"]}
        feat_names.update(self.paths["custom_paths"])
        features = {}
        for expert, rel_names in feat_names.items():
            if expert not in self.ordered_experts:
                continue
            feat_paths = tuple([Path(root_feat) / rel_name for rel_name in rel_names])
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
            self.text_features = memcache(root_feat /
                                          self.paths["text_feat_paths"]["train"])
            self.text_features.update(
                memcache(root_feat / self.paths["text_feat_paths"][self.split_name]))
            self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for ANet, since we assume"
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
