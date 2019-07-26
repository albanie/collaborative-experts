import time
from os.path import join as pjoin
from pathlib import Path
from utils.util import memcache
from base.base_dataset import BaseDataset


class DiDeMo(BaseDataset):

    def configure_train_test_splits(self, split_name):
        train_list_path = "train_list.txt"
        if split_name == "val":
            test_list_path = "val_list.txt"
        elif split_name == "test":
            test_list_path = "test_list.txt"
        else:
            raise ValueError(f"Unrecognised DiDeMo split: {split_name}")

        print("loading training/val splits....")
        tic = time.time()
        with open(pjoin(self.root_feat, train_list_path)) as f:
            self.train_list = f.readlines()
        with open(pjoin(self.root_feat, test_list_path)) as f:
            self.test_list = f.readlines()
        
        # For DiDeMo, we need to remove additional video suffixes
        self.train_list = [x.strip().split(".")[0] for x in self.train_list]
        self.test_list = [x.strip().split(".")[0] for x in self.test_list]

        print("done in {:.3f}s".format(time.time() - tic))
        self.split_name = split_name

    def load_features(self):
        root_feat = self.root_feat
        feat_names = {
            "face": "VGGFace2-ResNet50-face-avg.pickle",
            "flow": "i3d-i3d-avg.pickle",
            "rgb": f"{self.rgb_model_name}-imagenet-avg.pickle",
            "scene": "densenet161-scene-max.pickle",
            "ocr": "ocr-feats.pkl",
            "audio": "vggish-audio-raw.pickle",
            "speech": "stt_w2v.pickle",
        }
        feat_paths = {key: Path(root_feat) / value for key, value in feat_names.items()}

        if self.text_feat == "openai":
            text_feat_path = pjoin(root_feat, "openai-feats.pkl")
        else:
            raise ValueError(f"Text features {self.text_feat} not supported ")

        features = {expert: memcache(path) for expert, path in feat_paths.items()}
        text_features = memcache(text_feat_path)
        self.features = features
        self.text_features = text_features
        self.raw_captions = memcache(Path(self.data_dir) / "processing/raw-captions.pkl")

    def sanity_checks(self):
        msg = (f"Expected to have single test caption for DiDemo, since we assume"
               f"that the captions are fused (but using {self.num_test_captions})")
        assert self.num_test_captions == 1, msg
