import copy
from typing import Dict, Union, List
from pathlib import Path

from typeguard import typechecked
from zsvision.zs_utils import memcache, concat_features

from utils import memory_summary
from base.base_dataset import BaseDataset


class DiDeMo(BaseDataset):

    @staticmethod
    @typechecked
    def dataset_paths() -> Dict[str, Union[Path, str, Dict, List[str]]]:
        subset_paths = {}
        test_splits = {
            "val": "val_list.txt",
            "test": "test_list.txt",
            "public_server_val": "public_server_val.txt",
            "public_server_test": "public_server_test.txt",
        }
        for split_name, fname in test_splits.items():
            subset_paths[split_name] = {"train": "train_list.txt", "val": fname}

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
            "audio": ["aggregated_audio_feats/vggish-audio-raw.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-feats.pkl"],
            "speech": ["aggregated_speech_feats/stt_w2v.pickle"],
            "face": ["aggregated_facefeats_25fps_256px_stride1/face-avg.pickle"],
        }
        text_feat_paths = {"openai": "openai-feats.pkl"}
        text_feat_paths = {key: Path("aggregated_text_feats") / fname
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
            text_feat_path = root_feat / self.paths["text_feat_path"][self.text_feat]
            self.text_features = memcache(text_feat_path)

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for DiDemo, since we assume"
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
