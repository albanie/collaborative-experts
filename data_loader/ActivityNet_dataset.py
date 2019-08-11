import time
from os.path import join as pjoin
from pathlib import Path
from utils.util import memcache
from base.base_dataset import BaseDataset


class ActivityNet(BaseDataset):

    def configure_train_test_splits(self, split_name):
        train_list_path = "train_list.txt"
        if split_name == "val1":
            test_list_path = "val_1_list.txt"
            raw_caps_name = "raw-captions-train-val_1.pkl"
        elif split_name == "val2":
            test_list_path = "val_2_list.txt"
            raw_caps_name = "raw-captions-train-val_2.pkl"
        else:
            raise ValueError(f"Unrecognised activity-net split: {split_name}")

        train_list_path = pjoin(self.root_feat, train_list_path)
        test_list_path = pjoin(self.root_feat, test_list_path)
        self.raw_captions_path = Path(self.root_feat) / raw_caps_name

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
            "face": "VGGFace2-ResNet50-face-avg.pickle",
            "flow": "i3d-i3d-avg.pickle",
            "rgb": f"{self.rgb_model_name}-imagenet-avg-nocrop.pickle",
            "scene": "densenet161-scene-max.pickle",
            "ocr": "AN_OCR_ALL_unique_video_w2v.pkl",
            "audio": "vggish-audio-raw.pickle",
            "speech": "stt_w2v.pickle",
        }
        feat_paths = {key: Path(root_feat) / value for key, value in feat_names.items()}

        if self.text_feat == "openai":
            text_feat_train_path = pjoin(root_feat, "openai-train.pkl")
            text_feat_val1_path = pjoin(root_feat, "openai-val1.pkl")
            text_feat_val2_path = pjoin(root_feat, "openai-val2.pkl")
        else:
            raise ValueError(f"Text features {self.text_feat} not supported ")

        features = {expert: memcache(path) for expert, path in feat_paths.items()}
        text_features = memcache(text_feat_train_path)
        if self.split_name == "val1":
            text_features.update(memcache(text_feat_val1_path))
        elif self.split_name == "val2":
            text_features.update(memcache(text_feat_val2_path))
        else:
            raise ValueError(f"unrecognised activity-net split: {self.split_name}")

        self.features = features
        self.text_features = text_features
        self.raw_captions = memcache(self.raw_captions_path)

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for ANet, since we assume"
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
