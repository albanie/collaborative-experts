import copy
from pathlib import Path

from utils import memory_summary
from utils.util import concat_features
from zsvision.zs_utils import memcache
from base.base_dataset import BaseDataset


class ActivityNet(BaseDataset):

    @staticmethod
    def dataset_paths(split_name, text_feat):
        train_list_path = "train_list.txt"
        if split_name == "val1":
            test_list_path = "val_1_list.txt"
            raw_caps_name = "raw-captions-train-val_1.pkl"
        elif split_name == "val2":
            test_list_path = "val_2_list.txt"
            raw_caps_name = "raw-captions-train-val_2.pkl"
        else:
            raise ValueError(f"Unrecognised activity-net split: {split_name}")
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
        custom_paths = {
            "audio": ["aggregated_audio/vggish-audio-raw.pickle"],
            "speech": ["aggregated_speech/goog_w2v-speech-raw.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-w2v.pkl"],
            "face": ["aggregated_facefeats_25fps_256px_stride1/face-avg.pickle"],
        }
        text_feat_names = {key: f"{text_feat}-{key}"
                           for key in {"train", "val1", "val2"}}
        text_feat_paths = {key: f"aggregated_text_feats/{val}.pkl"
                           for key, val in text_feat_names.items()}
        feature_info = {
            "subset_list_paths": subset_paths,
            "feature_names": feature_names,
            "custom_paths": custom_paths,
            "text_feat_paths": text_feat_paths,
            "raw_captions_path": raw_caps_name,
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
        self.text_features = memcache(root_feat / self.paths["text_feat_paths"]["train"])
        self.text_features.update(
            memcache(root_feat / self.paths["text_feat_paths"][self.split_name]))
        self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for ANet, since we assume"
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
