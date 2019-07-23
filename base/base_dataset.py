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
        self.raw_captions = memcache(Path(data_dir) / "processing/raw-captions.pkl")
        self.rgb_shots = 1
        self.experts = set(raw_input_dims.keys())

        print("USING SINGLE CAPTION PER TRAINING VIDEO")
        self.captions_per_video = 1

        # TODO(Samuel) - is a global fixed ordering still necessary?
        self.ordered_experts = list(raw_input_dims.keys())

        self.configure_train_test_splits(split_name=split_name)
        self.num_train = len(self.train_list)
        self.raw_input_dims = raw_input_dims

        # we store paths to enable visualisations
        video_paths = [Path(data_dir) / f"videos/{x}.mp4" for x in self.test_list]
        self.video_path_retrieval = video_paths

        # NOTE: We use nans rather than zeros to indicate missing faces
        self.MISSING_VAL = np.nan
        self.load_features()
        num_test = len(self.test_list)

        # tensors are allocated differently, depending on whether they are expected to
        # vary in size
        fixed_sz_experts = {"face", "flow", "scene", "rgb"}
        variable_sz_experts = {"audio", "speech", "ocr"}

        # we only allocate storage for experts used by the current dataset
        self.fixed_sz_experts = fixed_sz_experts.intersection(self.experts)
        self.variable_sz_experts = variable_sz_experts.intersection(self.experts)

        retrieval = {expert: np.zeros((num_test, max_words, raw_input_dims[expert]))
                     for expert in self.variable_sz_experts}

        # The rgb modality is handled separately from the others because its shape can
        # vary along a # different dimension (according to whether or not dynamic shot
        # boundaries are used)
        retrieval.update({expert: np.zeros((num_test, raw_input_dims[expert]))
                          for expert in self.fixed_sz_experts if expert != "rgb"})
        retrieval["rgb"] = np.zeros((num_test, self.rgb_shots, raw_input_dims["rgb"]))
        self.retrieval = retrieval
        self.text_retrieval = np.zeros((num_test, self.num_test_captions,
                                        max_words, self.text_dim))

        # some "flaky" experts are only available for a fraction of videos - we need
        # to pass this information (in the form of indices) into the network for any
        # experts present in the current dataset
        flaky_experts = {"face", "audio", "speech", "ocr", "flow"}
        self.flaky_experts = flaky_experts.intersection(self.experts)
        self.test_ind = {expert: th.ones(num_test) for expert in self.experts}
        self.raw_captions_retrieval = [None] * num_test

        # avoid evaluation on missing queries
        self.query_masks = np.zeros((num_test, num_test_captions))
        for ii, video_name in enumerate(self.test_list):

            self.raw_captions_retrieval[ii] = self.raw_captions[video_name]

            for expert in self.fixed_sz_experts.intersection(self.experts):
                self.retrieval[expert][ii] = self.aggregate_feats(
                    feats=self.features[expert][video_name],
                    video_name=video_name,
                    modality=expert,
                )
            for expert in self.variable_sz_experts.intersection(self.experts):
                keep = min(max_words, len(self.features[expert][video_name]))
                feats = self.features[expert][video_name][: keep]
                self.retrieval[expert][ii, :keep, :] = feats

            # Since some missing faces are stored as zeros, rather than nans, we handle
            # their logic separately
            for expert in self.flaky_experts:
                if expert != "face":
                    if any(np.isnan(self.retrieval[expert][ii].flatten())):
                        self.test_ind[expert][ii] = 0
                else:
                    # logic for checking faces
                    face_feat = self.retrieval[expert][ii]
                    if np.array_equal(np.unique(face_feat), np.array([0])):
                        face_ind = 0
                    elif face_feat.ndim <= 2:
                        if face_feat.ndim == 1:
                            face_feat = face_feat.reshape(1, -1)
                        msg = "failure checking faces"
                        assert face_feat.shape[1] == self.raw_input_dims["face"], msg
                        face_ind = not self.has_missing_values(face_feat)
                    else:
                        raise ValueError(f"unexpected shape {face_feat.shape}")
                    self.test_ind[expert][ii] = face_ind

            candidates_sentences = self.text_features[video_name]

            if self.restrict_test_captions is not None:
                keep_sent_idx = self.restrict_test_captions[video_name]
                candidates_sentences = [candidates_sentences[keep_sent_idx]]

            self.query_masks[ii, :len(candidates_sentences)] = 1

            msg = "{}/{} Evaluating with sentence {} out of {} (has {} words) for {}"
            for test_caption_idx in range(self.num_test_captions):
                if len(candidates_sentences) <= test_caption_idx:
                    break
                keep = min(len(candidates_sentences[test_caption_idx]), max_words)
                if ii % 500 == 0:
                    print(msg.format(ii, len(self.test_list), test_caption_idx,
                          len(candidates_sentences), keep, video_name))
                text_feats = candidates_sentences[test_caption_idx][: keep]
                if text_feats.size == 0:
                    print("WARNING-WARNING: EMPTY TEXT FEATURES!")
                    text_feats = 0
                    import ipdb; ipdb.set_trace()
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
        batch_size = len(data)

        # Track which indices of each modality are available in the present batch
        ind = {expert: np.zeros(batch_size) for expert in self.experts}

        # as above, we handle rgb separately from other fixed_sz experts
        tensors = {expert: np.zeros((batch_size, self.raw_input_dims[expert]))
                   for expert in self.fixed_sz_experts if expert != "rgb"}
        tensors["rgb"] = np.zeros((batch_size, self.rgb_shots,
                                   self.raw_input_dims["rgb"]))
        tensors.update({expert: np.zeros(
            (batch_size, self.max_words, self.raw_input_dims[expert])
        ) for expert in self.variable_sz_experts})

        text_tensor = np.zeros((batch_size, self.captions_per_video, self.max_words,
                                self.text_dim))

        for ii, _ in enumerate(data):

            datum = data[ii]
            for expert in self.experts:
                ind[expert][ii] = datum[f"{expert}_ind"]
            ind = {key: ensure_tensor(val) for key, val in ind.items()}

            # It is preferable to explicitly pass NaNs into the network as missing
            # values, over simply zeros, to avoid silent failures
            for expert in self.fixed_sz_experts:
                tensors[expert][ii] = datum[expert]
            for expert in self.variable_sz_experts:
                keep = min(len(datum[expert]), self.max_words)
                if keep:
                    tensors[expert][ii, :keep, :] = datum[expert][:keep]
            text = datum["text"]
            for jj in range(self.captions_per_video):
                keep = min(len(text[jj]), self.max_words)
                text_tensor[ii, jj, :keep, :] = text[jj][:keep]

        text = th.from_numpy(text_tensor).float()
        experts = OrderedDict(
            (expert, th.from_numpy(tensors[expert]).float())
            for expert in self.ordered_experts)

        minibatch = {"text": text, "experts": experts, "ind": ind}

        # ----------------------------------------------------------------
        # sanity checking
        # ----------------------------------------------------------------
        if False:
            import pickle
            with open("/tmp/minibatch.pkl", "rb") as f:
                minibatch2 = pickle.load(f)

            # text - OK
            print("text diff", (minibatch2["text"] - minibatch["text"]).sum())

            # ind - OK
            for key, val in minibatch["ind"].items():
                print(f"ind diff: {key}", (val - minibatch2["ind"][key]).sum())

            # set nans to comparable number first
            NAN_VAL = 2780343
            experts1 = dict(minibatch["experts"])
            experts2 = dict(minibatch2["experts"])
            for key in experts1.keys():
                fix = th.isnan(experts1[key])
                experts1[key][fix] = NAN_VAL

                fix = th.isnan(experts2[key])
                experts2[key][fix] = NAN_VAL

            for key in experts1:
                print(key, (experts1[key] - experts2[key]).sum())
            import ipdb; ipdb.set_trace()
        return minibatch

    def __len__(self):
        return self.num_train

    def __getitem__(self, idx):
        if idx < self.num_train:
            vid = self.train_list[idx]
            text = self.text_features[vid]
            text = np.random.choice(text, size=self.captions_per_video)
            features = {expert: self.features[expert][vid] for expert in self.experts}

            # We need to handle face availablility separately, because some missing faces
            # are store simply as zero-vectors, rather than as NaNs.
            ind = {expert: not self.has_missing_values(features[expert]) for expert
                   in self.flaky_experts if expert != "face"}
            ind.update({expert: 1 for expert in self.experts if expert
                        not in self.flaky_experts})

            # logic for checking faces
            if np.array_equal(np.unique(features["face"]), np.array([0])):
                face_ind = 0
            elif features["face"].ndim > 1:
                msg = "failure checking faces"
                assert features["face"].shape[1] == self.raw_input_dims["face"], msg
                # face_feats = np.mean(features["face"], 0, keepdims=True)
                face_ind = not self.has_missing_values(features["face"])
            else:
                raise ValueError("unexpected size")
            ind["face"] = face_ind

            # NOTE: due to differences in how the features were stored, certain kinds
            # need to be aggregated along the temporal dimension
            msg = ("When features have not yet been aggregated, expect feature-dim"
                   "to lie on the second dimension")
            unaggregated = {"flow", "rgb", "scene", "face"}
            for expert in unaggregated:
                assert features[expert].ndim == 2
                assert features[expert].shape[1] == self.raw_input_dims[expert], msg
                features[expert] = self.aggregate_feats(
                    video_name=vid,
                    feats=features[expert],
                    modality=expert,
                )
        # Return both the missing indices as well as the tensors
        sample = {"text": text}
        sample.update({f"{key}_ind": val for key, val in ind.items()})
        sample.update(features)
        return sample

    def get_retrieval_data(self):
        experts = OrderedDict(
            (expert, th.from_numpy(self.retrieval[expert]).float())
            for expert in self.ordered_experts)

        retrieval_data = {
            "text": ensure_tensor(self.text_retrieval).float(),
            "experts": experts,
            "ind": self.test_ind,
        }
        meta = {
            "query_masks": self.query_masks,
            "raw_captions": self.raw_captions_retrieval,
            "paths": self.video_path_retrieval,
        }

        if False:
            # safety checks
            import pickle
            ret1 = retrieval_data
            with open("/tmp/retrieval_data.pkl", "rb") as f:
                ret2 = pickle.load(f)

            # ind - OK
            for key, val in ret1["ind"].items():
                print(f"ind diff: {key}", (val - ret2["ind"][key].float()).sum())

            # set nans to comparable number first
            NAN_VAL = 2780343
            experts1 = dict(ret1["experts"])
            experts2 = dict(ret2["experts"])
            for key in experts1.keys():
                fix = th.isnan(experts1[key])
                experts1[key][fix] = NAN_VAL

                fix = th.isnan(experts2[key])
                experts2[key][fix] = NAN_VAL

            for key in experts1:
                print(key, (experts1[key] - experts2[key]).sum())

            # text - OK
            print("text diff", (ret2["text"] - ret1["text"]).sum())
            import ipdb; ipdb.set_trace()
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
