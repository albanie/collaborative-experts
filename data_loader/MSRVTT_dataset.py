import time
from pathlib import Path
from os.path import join as pjoin
from utils.util import memcache
from base.base_dataset import BaseDataset


class MSRVTT(BaseDataset):

    def configure_train_test_splits(self, split_name):
        self.restrict_test_captions = None
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
            test_cap_idx_path = pjoin(self.root_feat, "jsfusion_val_caption_idx.pkl")
            self.restrict_test_captions = memcache(test_cap_idx_path)
        elif split_name in {"full-val", "full-test"}:
            train_list_path = "train_list_full.txt"
            if split_name == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(split_name))

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
        feat_paths = {}

        if self.split_name == "miech":
            if self.rgb_model_name == "resnet":
                rgb_feat_name = "resnet_features.pickle"
            elif self.rgb_model_name == "senet154":
                rgb_feat_name = "senet154-imagenet-raw-nocrop.pickle"
            else:
                raise ValueError(f"unrecognised rgb_model_name: {self.rgb_model_name}")
            feat_paths["audio"] = pjoin(root_feat, "audio_features.pickle")
            feat_paths["face"] = pjoin(root_feat, "face_features.pickle")
            feat_paths["flow"] = pjoin(root_feat, "flow_features.pickle")
        elif self.split_name in {"full-test", "full-val", "jsfusion"}:
            feat_paths["audio"] = pjoin(root_feat, "Audio_MSRVTT_new.pickle")
            feat_paths["face"] = pjoin(root_feat, "Face_MSRVTT_new.pickle")
            feat_paths["flow"] = pjoin(root_feat, "I3D_MSRVTT_new.pickle")
            rgb_feat_name = f"{self.rgb_model_name}-imagenet-raw-nocrop.pickle"

        feat_paths["rgb"] = pjoin(root_feat, rgb_feat_name)
        feat_paths["scene"] = pjoin(root_feat, "scene-raw.npy")

        # Note: Antoine's text features cover the full 10,000 videos, so can be
        # used for either split, similarly for the speech embeddings
        text_feat = self.text_feat
        if text_feat == "w2v":
            text_feat_path = pjoin(root_feat, "w2v_MSRVTT.pickle")
        elif text_feat == "openai":
            text_feat_path = pjoin(root_feat, "w2v_MSRVTT_openAIGPT.pickle")
        elif text_feat == "bertxl":
            text_feat_path = pjoin(root_feat, "w2v_MSRVTT_transformer.pickle")
        else:
            raise ValueError("Text features {} not recognised ".format(text_feat))
        feat_paths["speech"] = pjoin(root_feat, "stt_w2v.pickle")
        feat_paths["ocr"] = pjoin(root_feat, "MSR_VTT_all_text_w2v.pkl")
        # drop features which have not been requested
        feat_paths = {key: val for key, val in feat_paths.items()
                      if key in self.ordered_experts}
        features = {expert: memcache(path) for expert, path in feat_paths.items()}

        # we handle ocr separately from the other experts, for backwards compatibility
        canon_feats = {}
        for expert, feats in features.items():
            if expert != "ocr":
                canon_feats[expert] = self.canonical_features(feats)
            else:
                raw_dim = self.raw_input_dims[expert]
                canon_feats[expert] = self.canonical_features(feats, raw_dim=raw_dim)
        self.features = canon_feats
        self.raw_captions = memcache(Path(self.data_dir) / "processing/raw-captions.pkl")
        self.text_features = memcache(text_feat_path)

    def sanity_checks(self):
        if self.num_test_captions == 20:
            if len(self.test_list) == 2990:
                missing = 6
            elif len(self.test_list) == 1000:
                missing = 2
            elif len(self.test_list) == 497:
                missing = 0
            else:
                raise ValueError("unrecognised test set")
            msg = "Expected to find two missing queries in MSRVTT for full eval"
            assert self.query_masks.sum() == self.query_masks.size - missing, msg
