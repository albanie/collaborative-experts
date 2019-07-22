import time
import torch as th
from abc import abstractmethod
from torch.utils.data import Dataset
import numpy as np
from os.path import join as pjoin
from utils.util import ensure_tensor
from collections import OrderedDict
from utils.util import memcache
from pathlib import Path


class BaseDataset(Dataset):

    @abstractmethod
    def configure_train_test_splits(self, split_name):
        """Partition the datset into train/val/test splits.

        Args:
            split_name (str): the name of the split
        """
        raise NotImplementedError

    @abstractmethod
    def sanity_checks(self):
        """Run sanity checks on loaded data
        """
        raise NotImplementedError

    @abstractmethod
    def load_features(self):
        """Load features from disk
        """
        raise NotImplementedError

    def __init__(self, data_dir, feat_aggregation, raw_input_dims, num_test_captions,
                 split_name, text_dim, text_feat, rgb_model_name, max_words=30,
                 verbose=False):

        self.text_feat = text_feat
        self.text_dim = text_dim
        self.max_words = max_words
        self.num_test_captions = num_test_captions
        self.rgb_model_name = rgb_model_name
        self.restrict_test_captions = None
        self.feat_aggregation = feat_aggregation
        self.root_feat = Path(data_dir) / "symlinked-feats"
        raw_caption_path = Path(data_dir) / "processing/raw-captions.pkl"
        self.raw_captions = memcache(raw_caption_path, "pkl")
        self.captions_per_video = 1
        self.rgb_shots = 1

        self.configure_train_test_splits(split_name=split_name)
        self.num_train = len(self.train_list)

        self.rgb_dim = raw_input_dims["rgb"]
        self.flow_dim = raw_input_dims["flow"]
        self.audio_dim = raw_input_dims["audio"]
        self.speech_dim = raw_input_dims["speech"]
        self.ocr_dim = raw_input_dims["ocr"]
        self.face_dim = raw_input_dims["face"]
        self.scene_dim = raw_input_dims["scene"]

        # we store paths to enable visualisations
        video_paths = [Path(data_dir) / "videos/{}.mp4".format(x)
                       for x in self.test_list]
        self.video_path_retrieval = video_paths

        # NOTE: We use nans rather than zeros to indicate missing faces
        self.MISSING_VAL = np.nan
        self.load_features()

        num_test = len(self.test_list)
        self.rgb_retrieval = np.zeros((num_test, self.rgb_shots, self.rgb_dim))
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
        self.text_retrieval = np.zeros((num_test, self.num_test_captions,
                                        max_words, self.text_dim))

        # avoid evaluation on missing queries
        self.query_masks = np.zeros((num_test, num_test_captions))
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
            for test_caption_idx in range(self.num_test_captions):
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
        self.sanity_checks()

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
        return self.num_train

    def __getitem__(self, idx):
        if idx < self.num_train:
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
            ocr_ind = not self.has_missing_values(ocr_feats)
            audio_ind = not self.has_missing_values(audio_feats)
            speech_ind = not self.has_missing_values(speech_feats)

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
                        # be handled with the same pipeline as other features
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
