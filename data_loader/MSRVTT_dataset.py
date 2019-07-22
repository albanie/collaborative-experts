import time
from os.path import join as pjoin
from utils.util import memcache
from base.base_dataset import BaseDataset


class MSRVTT(BaseDataset):
    def __init__(self, data_dir, feat_aggregation, raw_input_dims, num_test_captions,
                 split_name, text_dim, text_feat, rgb_model_name, max_words=30,
                 verbose=False):

        super().__init__(
            data_dir=data_dir,
            feat_aggregation=feat_aggregation,
            raw_input_dims=raw_input_dims,
            num_test_captions=num_test_captions,
            split_name=split_name,
            text_dim=text_dim,
            text_feat=text_feat,
            rgb_model_name=rgb_model_name,
            max_words=max_words,
            verbose=verbose,
        )

    def configure_train_test_splits(self, split_name):
        self.restrict_test_captions = None
        if split_name == "miech":
            train_list_path = "train_list.txt"
            test_list_path = "test_list.txt"
        elif split_name in "jsfusion":
            train_list_path = "jsfusion_train_list.txt"
            test_list_path = "jsfusion_val_list.txt"
            # NOTE: The JSFusion split (referred to as 1k-A in the paper) uses all
            # videos, but randomly samples a single caption per video from the test
            # set for evaluation. To reproduce this evaluation, we use the indices
            # of the test captions, and restrict to this subset during eval.
            test_cap_idx_path = pjoin(self.root_feat, "jsfusion_val_caption_idx.pkl")
            self.restrict_test_captions = memcache(test_cap_idx_path, "pkl")
        elif split_name in {"full-val", "full-val"}:
            train_list_path = "train_list_dev.txt"
            if split_name == "full-val":
                test_list_path = "validate_list_dev.txt"
            else:
                test_list_path = "test_list_new.txt"
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
        if self.split_name == "miech":
            if self.rgb_model_name == "resnet":
                rgb_feat_name = "resnet_features.pickle"
            elif self.rgb_model_name == "senet154":
                rgb_feat_name = "senet154-imagenet-raw-nocrop.pickle"
            else:
                raise ValueError(f"unrecognised rgb_model_name: {self.rgb_model_name}")
            audio_feat_path = pjoin(root_feat, "audio_features.pickle")
            face_feat_path = pjoin(root_feat, "face_features.pickle")
            flow_feat_path = pjoin(root_feat, "flow_features.pickle")
        elif self.split_name in {"full-test", "full-val", "jsfusion"}:
            audio_feat_path = pjoin(root_feat, "Audio_MSRVTT_new.pickle")
            face_feat_path = pjoin(root_feat, "Face_MSRVTT_new.pickle")
            flow_feat_path = pjoin(root_feat, "I3D_MSRVTT_new.pickle")
            rgb_feat_name = f"{self.rgb_model_name}-imagenet-raw-nocrop.pickle"
        rgb_feat_path = pjoin(root_feat, rgb_feat_name)
        scene_feat_path = pjoin(root_feat, "scene-raw.npy")

        # Note: Antoine's text features cover the full 10,000 videos, so can be
        # used for either split, similarly for the speech embeddings
        yang_dir = "/scratch/shared/slow/yangl/code/BERT/"
        text_feat = self.text_feat
        if text_feat == "w2v":
            text_feat_path = pjoin(root_feat, "w2v_MSRVTT.pickle")
        elif text_feat == "openai":
            text_feat_path = pjoin(yang_dir, "w2v_MSRVTT_openAIGPT.pickle")
        elif text_feat == "bertxl":
            text_feat_path = pjoin(yang_dir, "w2v_MSRVTT_transformer.pickle")
        else:
            raise ValueError("Text features {} not recognised ".format(text_feat))
        speech_feat_path = pjoin(root_feat, "stt_w2v.pickle")
        ocr_feat_path = pjoin(root_feat, "MSR_VTT_all_text_w2v.pkl")

        text_features = memcache(text_feat_path, "pkl")
        speech_features = memcache(speech_feat_path, "pkl")
        ocr_features = memcache(ocr_feat_path, "pkl")
        rgb_features = memcache(rgb_feat_path, "pkl")
        audio_features = memcache(audio_feat_path, "pkl")
        face_features = memcache(face_feat_path, "pkl")
        flow_features = memcache(flow_feat_path, "pkl")
        scene_features = memcache(scene_feat_path, "npy")

        self.audio_features = self.canonical_features(audio_features)
        self.face_features = self.canonical_features(face_features)
        self.flow_features = self.canonical_features(flow_features)
        self.scene_features = self.canonical_features(scene_features)
        self.speech_features = self.canonical_features(speech_features)
        self.ocr_features = self.canonical_features(ocr_features, raw_dim=self.ocr_dim)
        self.rgb_features = self.canonical_features(rgb_features)
        self.text_features = text_features
        """For now, we follow Antoine's approach of using the first text caption
        for the retreival task when evaluating on his custom split."""

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
