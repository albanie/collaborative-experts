import copy
import time
from pathlib import Path

from utils import memory_summary
from utils.util import memcache, ensure_tensor, concat_features
from base.base_dataset import BaseDataset


class DiDeMo(BaseDataset):

    @staticmethod
    def dataset_paths(split_name, text_feat):
        train_list_path = "train_list.txt"
        if split_name == "val":
            test_list_path = "val_list.txt"
        elif split_name == "test":
            test_list_path = "test_list.txt"
        else:
            raise ValueError(f"Unrecognised DiDeMo split: {split_name}")
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
            "audio": ["aggregated_audio_feats/vggish-audio-raw.pickle"],
            "ocr": ["aggregated_ocr_feats/ocr-feats.pkl"],
            "flow": ["antoine/i3d-i3d-max-fps25-stride25.pickle"],
            "speech": ["aggregated_speech_feats/stt_w2v.pickle"],
            "face": ["aggregated_facefeats_25fps_256px_stride1/face-avg.pickle"],
        }
        if text_feat == "openai":
            text_feat_name = "openai-feats.pkl"
        elif text_feat == "w2v":
            text_feat_name = "w2v-text.pickle"
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
        self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
        self.text_features = memcache(root_feat / self.paths["text_feat_path"])
        # self.raw_captions = memcache(Path(self.root_feat) / "raw-captions.pkl")
        # self.text_features = memcache(text_feat_path)

    # def configure_train_test_splits(self, split_name):
    #     print("loading training/val splits....")
    #     tic = time.time()
        # for subset, path in zip(["train", "val"], [train_list_path, test_list_path]):
        #     subset_list_path = Path(self.root_feat) / path
        #     with open(subset_list_path) as f:
        #         rows = f.read().splitlines()
        #         # For DiDeMo, we need to remove additional video suffixes
        #         self.partition_lists[subset] = [x.strip().split(".")[0] for x in rows]
        # print("done in {:.3f}s".format(time.time() - tic))
        # self.split_name = split_name

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for DiDemo, since we assume"
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
