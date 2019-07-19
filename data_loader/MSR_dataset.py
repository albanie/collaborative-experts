import time
import torch as th
from torch.utils.data import Dataset
import numpy as np
from os.path import join as pjoin
from utils.util import ensure_tensor
from collections import OrderedDict
from utils.util import memcache
from pathlib import Path


class MSRVTT_new(Dataset):
    """LSMDC dataset."""

    def __init__(self, data_dir, feat_aggregation, raw_input_dims, num_test_captions,
                 split_name, text_dim, text_feat, rgb_model_name, max_words=30,
                 verbose=False):

        root_feat = Path(data_dir) / "symlinked-feats"
        restrict_test_captions = None
        if split_name == "miech":
            train_list_path = "train_list.txt"
            test_list_path = "test_list.txt"
        elif split_name in "official":
            train_list_path = "train_list_new.txt"
            test_list_path = "test_list_new.txt"
        elif split_name == "dev":
            train_list_path = "train_list_dev.txt"
            test_list_path = "validate_list_dev.txt"
        elif split_name in "jsfusion":
            train_list_path = "jsfusion_train_list.txt"
            test_list_path = "jsfusion_val_list.txt"
            test_cap_idx_path = pjoin(root_feat, "jsfusion_val_caption_idx.pkl")
            restrict_test_captions = memcache(test_cap_idx_path, "pkl")
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(split_name))

        train_list_path = pjoin(root_feat, train_list_path)
        test_list_path = pjoin(root_feat, test_list_path)

        if split_name == "miech":
            if rgb_model_name == "resnet":
                rgb_feat_name = "resnet_features.pickle"
            elif rgb_model_name == "senet154":
                rgb_feat_name = "senet154-imagenet-raw-nocrop.pickle"
            else:
                msg = "unrecognised rgb_model_name: {}"
                raise ValueError(msg.format(rgb_model_name))
            audio_feat_path = pjoin(root_feat, "audio_features.pickle")
            face_feat_path = pjoin(root_feat, "face_features.pickle")
            flow_feat_path = pjoin(root_feat, "flow_features.pickle")
        elif split_name in {"official", "dev", "jsfusion"}:
            # rgb_feat_path = pjoin(root_feat, "resnet_MSRVTT_new.pickle")
            audio_feat_path = pjoin(root_feat, "Audio_MSRVTT_new.pickle")
            face_feat_path = pjoin(root_feat, "Face_MSRVTT_new.pickle")
            flow_feat_path = pjoin(root_feat, "I3D_MSRVTT_new.pickle")
            rgb_feat_name = "{}-imagenet-raw-nocrop.pickle".format(rgb_model_name)
        rgb_feat_path = pjoin(root_feat, rgb_feat_name)

        scene_feat_path = pjoin(root_feat, "scene-raw.npy")
        raw_caption_path = Path(data_dir) / "processing/raw-captions.pkl"
        raw_captions = memcache(raw_caption_path, "pkl")

        # fetcher = lambda: pickle_loader(data_paths["raw-captions"])
        # raw_captions = check_cache(key="raw_captions", fetcher=fetcher)

        # Note: Antoines text features cover the full 10,000 videos, so can be
        # used for either split, similarly for the speech embeddings
        yang_dir = "/scratch/shared/slow/yangl/code/BERT/"
        if text_feat == "w2v":
            text_feat_path = pjoin(root_feat, "w2v_MSRVTT.pickle")
        elif text_feat == "openai":
            text_feat_path = pjoin(yang_dir, "w2v_MSRVTT_openAIGPT.pickle")
        elif text_feat == "bertxl":
            text_feat_path = pjoin(yang_dir, "w2v_MSRVTT_transformer.pickle")
        else:
            raise ValueError("Text features {} not recognised ".format(text_feat))
        text_features = memcache(text_feat_path, "pkl")
        # text_fetcher = lambda: pickle_loader(text_feat_path)

        speech_feat_path = pjoin(root_feat, "stt_w2v.pickle")
        speech_features = memcache(speech_feat_path, "pkl")
        # speech_fetcher = lambda: pickle_loader(speech_feat_path)
        # ocr_fetcher = lambda: pickle_loader(ocr_feat_path)
        # ocr_features = check_cache(key="ocr_features", fetcher=ocr_fetcher)
        ocr_feat_path = pjoin(root_feat, "MSR_VTT_all_text_w2v.pkl")
        ocr_features = memcache(ocr_feat_path, "pkl")

        # text_feat_key = "text_features-{}".format(text_feat)
        # text_features = check_cache(key=text_feat_key, fetcher=text_fetcher)
        # speech_features = check_cache(key="speech_features", fetcher=speech_fetcher)
        rgb_features = memcache(rgb_feat_path, "pkl")
        audio_features = memcache(audio_feat_path, "pkl")
        face_features = memcache(face_feat_path, "pkl")
        flow_features = memcache(flow_feat_path, "pkl")
        scene_features = memcache(scene_feat_path, "npy")

        # fetcher = lambda: pickle_loader(rgb_feat_path)
        # rgb_features = check_cache(key="rgb_features", fetcher=fetcher)
        # fetcher = lambda: pickle_loader(audio_feat_path)
        # audio_features = check_cache(key="audio_features", fetcher=fetcher)
        # fetcher = lambda: pickle_loader(face_feat_path)
        # face_features = check_cache(key="face_features", fetcher=fetcher)
        # fetcher = lambda: pickle_loader(flow_feat_path)
        # flow_features = check_cache(key="flow_features", fetcher=fetcher)

        # fetcher = lambda: np_loader(scene_feat_path, l2norm=False)
        # scene_features = check_cache(key="scene_features", fetcher=fetcher)

        self.max_words = max_words
        self.feat_aggregation = feat_aggregation

        # only used by JSFusion
        self.restrict_test_captions = restrict_test_captions

        print("loading training/val splits....")
        tic = time.time()
        with open(train_list_path) as f:
            self.train_list = f.readlines()
        self.train_list = [x.strip() for x in self.train_list]

        with open(test_list_path) as f:
            self.test_list = f.readlines()
        self.test_list = [x.strip() for x in self.test_list]
        print("done in {:.3f}s".format(time.time() - tic))

        # NOTE: We use nans rather than zeros to indicate missing faces
        self.MISSING_VAL = np.nan
        self.raw_captions = raw_captions
        self.rgb_dim = raw_input_dims["rgb"]
        self.flow_dim = raw_input_dims["flow"]
        self.audio_dim = raw_input_dims["audio"]
        self.speech_dim = raw_input_dims["speech"]
        self.ocr_dim = raw_input_dims["ocr"]
        self.face_dim = raw_input_dims["face"]
        self.scene_dim = raw_input_dims["scene"]

        self.audio_features = self.canonical_features(audio_features)
        self.face_features = self.canonical_features(face_features)
        self.flow_features = self.canonical_features(flow_features)
        self.scene_features = self.canonical_features(scene_features)
        self.speech_features = self.canonical_features(speech_features)
        self.ocr_features = self.canonical_features(ocr_features, raw_dim=self.ocr_dim)
        self.captions_per_video = 1
        self.rgb_features = self.canonical_features(rgb_features)
        self.text_features = text_features
        self.n_MSR = len(self.train_list)

        # computing retrieval
        # self.scene_dim = raw_input_dims["scene"]

        self.text_dim = text_dim
        num_test = len(self.test_list)
        self.rgb_shots = 1
        self.rgb_retrieval = np.zeros((num_test, self.rgb_shots, self.rgb_dim))

        # we store paths to enable visualisations
        video_paths = [Path(data_dir) / "videos/{}.mp4".format(x)
                       for x in self.test_list]
        self.video_path_retrieval = video_paths

        self.flow_retrieval = np.zeros((num_test, self.flow_dim))
        self.scene_retrieval = np.zeros((num_test, self.scene_dim))
        self.audio_retrieval = np.zeros((num_test, max_words, self.audio_dim))
        self.speech_retrieval = np.zeros((num_test, max_words, self.speech_dim))
        self.ocr_retrieval = np.zeros((num_test, max_words, self.ocr_dim))
        self.face_retrieval = np.zeros((num_test, self.face_dim))
        self.face_ind_retrieval = np.ones((num_test))
        self.speech_ind_retrieval = np.ones((num_test))
        self.audio_ind_retrieval = np.ones((num_test))
        self.ocr_ind_retrieval = np.ones((num_test))
        self.rgb_ind_retrieval = np.ones((num_test))
        self.scene_ind_retrieval = np.ones((num_test))
        self.flow_ind_retrieval = np.ones((num_test))
        self.raw_captions_retrieval = [None] * num_test

        """For now, we follow Antoine's approach of using the first text caption
        for the retreival task when evaluating on his custom split."""
        self.text_retrieval = np.zeros((num_test, num_test_captions,
                                        max_words, self.text_dim))

        # avoid evaluation on missing queries
        self.query_masks = np.zeros((num_test, num_test_captions))
        print("UNRESOLVED: What should max words for audio even mean - crop time?")

        for ii, video_name in enumerate(self.test_list):
            self.rgb_retrieval[ii] = self.aggregate_feats(
                feats=self.rgb_features[video_name],
                video_name=video_name,
                modality="rgb",
            )
            self.flow_retrieval[ii] = self.aggregate_feats(
                feats=self.flow_features[video_name],
                video_name=video_name,
                modality="flow",
            )
            self.scene_retrieval[ii] = self.aggregate_feats(
                feats=self.scene_features[video_name],
                video_name=video_name,
                modality="scene",
            )
            self.raw_captions_retrieval[ii] = self.raw_captions[video_name]

            if len(self.face_features[video_name]) > 0:
                face_feat_test = self.face_features[video_name]
                self.face_retrieval[ii] = np.mean(face_feat_test, 0, keepdims=True)
            else:
                import ipdb; ipdb.set_trace()  # NOQA

            keep = min(max_words, len(self.audio_features[video_name]))
            audio_feats = self.audio_features[video_name][: keep]
            self.audio_retrieval[ii, :keep, :] = audio_feats

            keep = min(max_words, len(self.speech_features[video_name]))
            speech_feats = self.speech_features[video_name][: keep]
            self.speech_retrieval[ii, :keep, :] = speech_feats

            keep = min(max_words, len(self.ocr_features[video_name]))
            ocr_feats = self.ocr_features[video_name][: keep]
            self.ocr_retrieval[ii, :keep, :] = ocr_feats

            if any(np.isnan(self.face_retrieval[ii])):
                self.face_ind_retrieval[ii] = 0
            if any(np.isnan(self.speech_retrieval[ii].flatten())):
                self.speech_ind_retrieval[ii] = 0
            if any(np.isnan(self.ocr_retrieval[ii].flatten())):
                self.ocr_ind_retrieval[ii] = 0
            if any(np.isnan(self.audio_retrieval[ii].flatten())):
                self.audio_ind_retrieval[ii] = 0

            candidates_sentences = self.text_features[video_name]
            if self.restrict_test_captions is not None:
                keep_sent_idx = self.restrict_test_captions[video_name]
                candidates_sentences = [candidates_sentences[keep_sent_idx]]

            self.query_masks[ii, :len(candidates_sentences)] = 1
            msg = "{}/{} Evaluating with sentence {} out of {} (has {} words) for {}"
            for test_caption_idx in range(num_test_captions):
                if len(candidates_sentences) <= test_caption_idx:
                    # two MSRVTT videos only have 19 captions, so these use zeros at
                    # test time
                    # TODO(Samuel) - mask these values properly
                    assert len(candidates_sentences) == 19, "unexpected edge case"
                    continue
                keep = min(len(candidates_sentences[test_caption_idx]), max_words)
                if ii % 500 == 0:
                    print(msg.format(ii, len(self.test_list), test_caption_idx,
                          len(candidates_sentences), keep, video_name))
                text_feats = candidates_sentences[test_caption_idx][: keep]
                self.text_retrieval[ii, test_caption_idx, :keep, :] = text_feats

            # lt = len(self.text_features[self.test_list[i]])
            # self.text_retrieval[i, : min(max_words, lt), :] = \
            # self.text_features[self.test_list[i]][: min(max_words, lt)]
        if num_test_captions == 20:
            if len(self.test_list) == 2990:
                missing = 6
            elif len(self.test_list) == 1000:
                missing = 2
            elif len(self.test_list) == 497:
                missing = 0
            else:
                raise ValueError("unrecognised test set")
            msg = "Expected to find two missing queries in MSRVTT for official eval"
            assert self.query_masks.sum() == self.query_masks.size - missing, msg

        # self.rgb_retrieval = th.from_numpy(self.rgb_retrieval).float()
        # self.flow_retrieval = th.from_numpy(self.flow_retrieval).float()
        # self.scene_retrieval = th.from_numpy(self.scene_retrieval).float()
        # self.audio_retrieval = th.from_numpy(self.audio_retrieval).float()
        # self.speech_retrieval = th.from_numpy(self.speech_retrieval).float()
        # self.ocr_retrieval = th.from_numpy(self.ocr_retrieval).float()
        # self.face_retrieval = th.from_numpy(self.face_retrieval).float()
        # self.text_retrieval = th.from_numpy(self.text_retrieval).float()
        # self.query_masks = self.query_masks
        print("done")

    def aggregate_feats(self, feats, video_name, modality):
        mode = self.feat_aggregation[modality]
        if mode == "avg":
            agg = np.mean(feats, axis=0, keepdims=True)
        elif mode == "max":
            agg = np.max(feats, axis=0, keepdims=True)
        else:
            msg = "aggregation mode {} not supported"
            raise NotImplementedError(msg.format(mode))
        return agg

    def collate_data(self, data):
        face_ind = np.zeros((len(data)))
        speech_ind = np.zeros((len(data)))
        audio_ind = np.zeros((len(data)))
        ocr_ind = np.zeros((len(data)))
        rgb_ind = np.zeros((len(data)))
        scene_ind = np.zeros((len(data)))
        flow_ind = np.zeros((len(data)))
        rgb_tensor = np.zeros((len(data), self.rgb_shots, self.rgb_dim))
        flow_tensor = np.zeros((len(data), self.flow_dim))
        scene_tensor = np.zeros((len(data), self.scene_dim))
        face_tensor = np.zeros((len(data), self.face_dim))
        audio_tensor = np.zeros((len(data), self.max_words, self.audio_dim))
        speech_tensor = np.zeros((len(data), self.max_words, self.speech_dim))
        ocr_tensor = np.zeros((len(data), self.max_words, self.ocr_dim))
        text_tensor = np.zeros((len(data), self.captions_per_video, self.max_words,
                                self.text_dim))

        for ii, _ in enumerate(data):

            datum = data[ii]
            face_ind[ii] = datum["face_ind"]
            speech_ind[ii] = datum["speech_ind"]
            audio_ind[ii] = datum["audio_ind"]
            ocr_ind[ii] = datum["ocr_ind"]
            scene_ind[ii] = datum["scene_ind"]
            flow_ind[ii] = datum["flow_ind"]
            rgb_ind[ii] = datum["rgb_ind"]
            flow_tensor[ii] = datum["flow"]
            scene_tensor[ii] = datum["scene"]
            rgb_tensor[ii] = datum["rgb"]

            # if len(datum["face"]) > 0:
            # It is preferable to explicitly pass NaNs into the network as missing
            # values, over simply zeros, to avoid silent failures
            face_tensor[ii] = datum["face"]

            keep = min(len(datum["audio"]), self.max_words)
            audio_tensor[ii, :keep, :] = datum["audio"][:keep]

            keep = min(len(datum["speech"]), self.max_words)
            if keep:
                #  NOTE that we will pass in zeros here for now
                speech_tensor[ii, :keep, :] = datum["speech"][:keep]

            keep = min(len(datum["ocr"]), self.max_words)
            if keep:
                #  NOTE that we will pass in zeros here for now
                ocr_tensor[ii, :keep, :] = datum["ocr"][:keep]

            text = datum["text"]

            for jj in range(self.captions_per_video):
                keep = min(len(text[jj]), self.max_words)
                text_tensor[ii, jj, :keep, :] = text[jj][:keep]

        text = th.from_numpy(text_tensor).float()
        experts = OrderedDict([
            ("face", th.from_numpy(face_tensor).float()),
            ("rgb", th.from_numpy(rgb_tensor).float()),
            ("flow", th.from_numpy(flow_tensor).float()),
            ("scene", th.from_numpy(scene_tensor).float()),
            ("audio", th.from_numpy(audio_tensor).float()),
            ("speech", th.from_numpy(speech_tensor).float()),
            ("ocr", th.from_numpy(ocr_tensor).float()),
        ])
        indices = {
            "face": face_ind,
            "rgb": rgb_ind,
            "scene": scene_ind,
            "flow": flow_ind,
            "speech": speech_ind,
            "audio": audio_ind,
            "ocr": ocr_ind,
        }
        for key, val in indices.items():
            indices[key] = ensure_tensor(val)

        return {"text": text, "experts": experts, "ind": indices}

    def __len__(self):
        return self.n_MSR

    def __getitem__(self, idx):
        if idx < self.n_MSR:
            vid = self.train_list[idx]
            text = self.text_features[vid]

            text = np.random.choice(text, size=self.captions_per_video)

            audio_feats = self.audio_features[vid]
            flow_feats = self.flow_features[vid]
            scene_feats = self.scene_features[vid]
            face_feats = self.face_features[vid]
            rgb_feats = self.rgb_features[vid]
            speech_feats = self.speech_features[vid]
            ocr_feats = self.ocr_features[vid]

            rgb_ind = 1
            flow_ind = 1
            scene_ind = 1
            ocr_ind =  not self.has_missing_values(ocr_feats)
            audio_ind =  not self.has_missing_values(audio_feats)
            speech_ind =  not self.has_missing_values(speech_feats)

            # NOTE: due to differences in how the features were stored, certain kinds
            # need to be aggregated along the temporal dimension
            msg = ("When features have not yet been aggregated, expect feature-dim"
                   "to lie on the second dimension")
            assert flow_feats.ndim == 2
            assert flow_feats.shape[1] == self.flow_dim, msg
            flow_feats = np.mean(flow_feats, 0, keepdims=True)

            assert rgb_feats.ndim == 2
            assert rgb_feats.shape[1] == self.rgb_dim, msg
            rgb_feats = self.aggregate_feats(
                video_name=vid,
                feats=rgb_feats,
                modality="rgb",
            )
            # rgb_feats = np.mean(rgb_feats, 0, keepdims=True)

            assert scene_feats.ndim == 2
            assert scene_feats.shape[1] == self.scene_dim, msg
            scene_feats = np.mean(scene_feats, 0, keepdims=True)

            if np.array_equal(np.unique(face_feats), np.array([0])):
                face_ind = 0
            elif face_feats.ndim > 1:
                assert face_feats.shape[1] == self.face_dim, msg
                face_feats = np.mean(face_feats, 0, keepdims=True)
                if self.has_missing_values(face_feats):
                    face_ind = 0
                else:
                    face_ind = 1
            else:
                raise ValueError("unexpected size")

        return {
            "vid": vid,
            "text": text,
            "rgb": rgb_feats,
            "ocr": ocr_feats,
            "flow": flow_feats,
            "scene": scene_feats,
            "speech": speech_feats,
            "face": face_feats,
            "audio": audio_feats,
            "scene_ind": scene_ind,
            "flow_ind": flow_ind,
            "rgb_ind": rgb_ind,
            "face_ind": face_ind,
            "ocr_ind": ocr_ind,
            "speech_ind": speech_ind,
            "audio_ind": audio_ind,
        }

    def getRetrievalSamples(self):

        experts = OrderedDict([
            ("face", th.from_numpy(self.face_retrieval).float()),
            ("rgb", th.from_numpy(self.rgb_retrieval).float()),
            ("flow", th.from_numpy(self.flow_retrieval).float()),
            ("scene", th.from_numpy(self.scene_retrieval).float()),
            ("audio", th.from_numpy(self.audio_retrieval).float()),
            ("speech", th.from_numpy(self.speech_retrieval).float()),
            ("ocr", th.from_numpy(self.ocr_retrieval).float()),
        ])
        indices = {
            "face": self.face_ind_retrieval,
            "rgb": self.rgb_ind_retrieval,
            "scene": self.scene_ind_retrieval,
            "flow": self.flow_ind_retrieval,
            "speech": self.speech_ind_retrieval,
            "audio": self.audio_ind_retrieval,
            "ocr": self.ocr_ind_retrieval,
        }
        for key, val in indices.items():
            indices[key] = ensure_tensor(val)

        retrieval_data = {
            "text": ensure_tensor(self.text_retrieval).float(),
            "experts": experts,
            "ind": indices,
        }
        meta = {
            "query_masks": self.query_masks,
            "raw_captions": self.raw_captions_retrieval,
            "paths": self.video_path_retrieval,
        }
        return retrieval_data, meta
        # return {
        #     "flow": self.flow_retrieval,
        #     "face": self.face_retrieval,
        #     "rgb": self.rgb_retrieval,
        #     "audio": self.audio_retrieval,
        #     "speech": self.speech_retrieval,
        #     "ocr": self.ocr_retrieval,
        #     "scene": self.scene_retrieval,
        #     "paths": self.video_path_retrieval,
        #     "raw_captions": self.raw_captions_retrieval,
        #     "face_ind": self.face_ind_retrieval,
        #     "speech_ind": self.speech_ind_retrieval,
        #     "audio_ind": self.audio_ind_retrieval,
        #     "ocr_ind": self.ocr_ind_retrieval,
        #     "query_masks": self.query_masks,
        # }

    def canonical_features(self, x, raw_dim=None, keep_zeros=False):
        """Precomputed features should have the same format prior to aggregation.  The
        first dimension is temporal, the second dimension is the feature-dim, unless
        VLAD aggregation is to be used across instances, in which case the second dim
        is the instance dim and the third dim is the feature dim.

        For certain kinds of features, e.g. audio, the zero features must be kept in
        place - this is enabled by `keep_zeros`.
        """
        print("mappinig to canonical features....")
        tic = time.time()
        canonical_feats = {}
        for key, value in x.items():
            value = value.astype(np.float32)  # avoid overflow
            only_zeros = np.sum(value) == 0
            missing_dim = value.shape[0] == 0
            if (only_zeros and not keep_zeros) or missing_dim:
                if missing_dim:
                    if value.ndim != 2 and raw_dim:
                        # This is added for backwards compat. so that ocr features can
                        # be handled with the same pipelin as other features
                        value = np.zeros((0, raw_dim))
                    msg = "expected empty feats to have 2 dims or be fully empty"
                    assert value.ndim == 2, msg
                value = np.empty((1, value.shape[1]))
                value[:] = self.MISSING_VAL
            elif value.ndim == 1:
                value = value.reshape((-1, value.size))
            elif value.ndim > 2:
                raise ValueError("unexpected value shape: {}".format(value.shape))
            elif value.ndim == 2:
                assert value.shape[0] > 0
                assert value.shape[1] > 0
            canonical_feats[key] = value
        print("done in {:.3f}s".format(time.time() - tic))
        return canonical_feats

    def has_missing_values(self, x):
        # We check the first value to look for a missing feature marker (checking
        # the whole array would slow things down and isn't necessary)
        assert x.ndim == 2, "expected to check matrix"
        # NOTE: we cannot check equality against a NaN
        return np.isnan(x[0][0])
