import time
from os.path import join as pjoin
from pathlib import Path
from utils.util import memcache
from base.base_dataset import BaseDataset


class MSVD(BaseDataset):

    def configure_train_test_splits(self, split_name):
        train_list_path = "train_list.txt"
        if split_name == "official":
            test_list_path = "test_list.txt"
        elif split_name == "dev":
            test_list_path = "val_list.txt"
        else:
            raise ValueError(f"unrecognised MSVD split: {split_name}")

        train_list_path = pjoin(self.root_feat, train_list_path)
        test_list_path = pjoin(self.root_feat, test_list_path)

        print("loading training/val splits....")
        tic = time.time()
        with open(train_list_path) as f:
            self.train_list = f.readlines()
        self.train_list = [x.strip() for x in self.train_list]
        with open(test_list_path) as f:
            self.test_list = f.readlines()
        self.test_list = [x.strip() for x in self.test_list]
        print("done in {:.3f}s".format(time.time() - tic))
        self.split_name = split_name

    def load_features(self):
        root_feat = self.root_feat
        feat_names = {
            "face": "VGGFace2-ResNet50-face-raw.pickle",
            "flow": "i3d-i3d-raw.pickle",
            "rgb": f"{self.rgb_model_name}-imagenet-raw-nocrop.pickle",
            "scene": "densenet161-scene-max.pickle",
            "ocr": "MSVD_all_text_w2v.pkl",
        }
        feat_paths = {key: Path(root_feat) / value for key, value in feat_names.items()}

        if self.text_feat == "w2v":
            text_feat_train_path = pjoin(root_feat, "w2v-caption-train.pkl")
            text_feat_val_path = pjoin(root_feat, "w2v-caption-val.pkl")
            text_feat_test_path = pjoin(root_feat, "w2v-caption-test.pkl")
        elif self.text_feat == "openai":
            text_feat_train_path = pjoin(root_feat, "openai-caption-train.pkl")
            text_feat_val_path = pjoin(root_feat, "openai-caption-val.pkl")
            text_feat_test_path = pjoin(root_feat, "openai-caption-test.pkl")
        else:
            raise ValueError(f"Text features {self.text_feat} not supported ")

        features = {expert: memcache(path) for expert, path in feat_paths.items()}
        text_features = memcache(text_feat_train_path)
        if self.split_name == "dev":
            text_features.update(memcache(text_feat_val_path))
        elif self.split_name == "official":
            text_features.update(memcache(text_feat_test_path))
        else:
            raise ValueError(f"unrecognised MSVD split: {self.split_name}")

        # To ensure that the text features are stored with the same keys as other
        # features, we need to convert text feature keys (YouTube hashes) into
        # video names
        key_map = memcache(pjoin(root_feat, "dict_youtube_mapping.pkl"))
        inverse_map = {}
        for key, value in key_map.items():
            inverse_map[value] = key
        text_features = {inverse_map[key]: val for key, val in text_features.items()}

        # we handle ocr separately from the other experts, for backwards compatibility
        # reasons
        canon_feats = {}
        for expert, feats in features.items():
            if expert != "ocr":
                canon_feats[expert] = self.canonical_features(feats)
            else:
                raw_dim = self.raw_input_dims[expert]
                canon_feats[expert] = self.canonical_features(feats, raw_dim=raw_dim)
        self.features = canon_feats
        self.text_features = text_features
        self.raw_captions = memcache(Path(self.data_dir) / "processing/raw-captions.pkl")

    def sanity_checks(self):
        assert self.num_test_captions == 81, "Expected to have 81 test caps for MSVD"
