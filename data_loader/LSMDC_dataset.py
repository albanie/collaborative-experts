""" LSMDC dataset module.

NOTE: This code is loosely based on the LSMDC dataset loader provided in the MoEE
codebase:
https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/LSMDC.py
"""

import torch as th
import numpy as np
from collections import OrderedDict
from utils.util import ensure_tensor
from torch.utils.data import Dataset
from utils.util import memcache
from pathlib import Path


class LSMDC(Dataset):

    def __init__(self, data_dir, feat_aggregation, raw_input_dims, num_test_captions,
                 split_name, text_dim, text_feat, rgb_model_name, fuse_captions,
                 max_text_words, max_expert_tokens, verbose=False):

        self.ordered_experts = list(raw_input_dims.keys())
        self.max_expert_tokens = max_expert_tokens
        self.max_text_words = max_text_words
        self.raw_input_dims = raw_input_dims
        self.captions_per_video = 1
        self.MISSING_VAL = np.nan

        root_feat = Path(data_dir) / "symlinked-feats"

        print("Reading test data ...")
        train_feat_names = {
            "face": "X_face.npy",
            "flow": "X_flow.npy",
            "rgb": "X_resnet.npy",
            "scene": f"densenet161-scene-{feat_aggregation['scene']}-train.npy",
            "ocr": "w2v-ocr-raw-train.npy",
            "audio": "X_audio_train.npy",
        }
        val_feat_names = {
            "face": "face-retrieval.npy.tensor.npy",
            "flow": "flow-retrieval.npy.tensor.npy",
            "rgb": "resnet152-retrieval.npy.tensor.npy",
            "scene": f"densenet161-scene-{feat_aggregation['scene']}-val.npy",
            "ocr": "w2v-ocr-raw-val.npy",
            "audio": "X_audio_retrieval.npy.tensor.npy",
        }
        feat_paths = {"train": train_feat_names, "val": val_feat_names}

        if text_feat == "w2v":
            text_train = "w2v_LSMDC.npy"
            text_val = "w2v_LSMDC_retrieval.npy"
        elif text_feat == "openai":
            text_train = "openai-train.npy"
            text_val = "openai-test.npy"
        else:
            raise ValueError(f"Text features {text_feat} not recognised ")
        text_paths = {"train": text_train, "val": text_val}

        features = {}
        for key, feat_names in feat_paths.items():
            features[key] = {expert: memcache(Path(root_feat) / path) for expert, path
                             in feat_names.items()}
        text_features = {key: memcache(Path(root_feat) / val)
                         for key, val in text_paths.items()}

        # There are five videos without captions in the training set, so we drop them
        expected = 5
        train_masks = np.array([len(x) > 0 for x in text_features["train"]])
        missing_captions = len(train_masks) - sum(train_masks)
        msg = f"Expected {expected} videos without captions, found {missing_captions}"
        assert missing_captions == expected, msg
        features["train"] = {key: val[train_masks] for key, val
                             in features["train"].items()}
        with open(Path(root_feat) / "test_video_paths.txt", "r") as f:
            self.video_path_retrieval = [Path(x) for x in f.read().splitlines()]

        # combine variable length inputs into a large single tensor by zero padding. We
        # store the original sizes to allow reduced padding in minibatches
        self.expert_feat_sizes = {}
        for expert in {"audio", "ocr"}:
            feats = features["train"][expert]
            tensor, cropped_sizes = self.zero_pad_to_tensor(feats, self.max_expert_tokens)
            features["train"][expert] = tensor
            self.expert_feat_sizes[expert] = cropped_sizes

        text_features["train"] = text_features["train"][train_masks]
        self.text_feature_sizes = {}
        for key, val in text_features.items():
            tensor, cropped_sizes = self.zero_pad_to_tensor(val, self.max_text_words)
            self.text_feature_sizes[key], text_features[key] = cropped_sizes, tensor

        # store the indices of missing face and ocr features, marking the other experts
        # as available
        self.flaky = {"face", "ocr"}
        ind_paths = {x: Path(root_feat) / f"no_{x}_ind_retrieval.npy" for x in self.flaky}
        test_ind = {expert: 1 - memcache(path) for expert, path in ind_paths.items()}
        test_ind.update({expert: np.ones_like(test_ind["ocr"]) for expert in
                         self.ordered_experts if expert not in self.flaky})
        self.test_ind = {key:th.from_numpy(val) for key, val in test_ind.items()}

        for key in {"train", "val"}:
            missing = np.sum(features[key]["face"], axis=1) == 0
            features[key]["face"][missing, :] = np.nan
            missing = np.sum(np.sum(features[key]["ocr"], axis=1), axis=1) == 0
            features[key]["ocr"][missing, :] = np.nan

        self.features = features
        self.text_retrieval = th.from_numpy(text_features["val"]).float()
        self.raw_captions_retrieval = None
        self.text_features = text_features

    def __getitem__(self, idx):
        features = {expert: self.features["train"][expert][idx]
                    for expert in self.ordered_experts}
        ind = {expert: 1 for expert in self.ordered_experts}
        for expert in self.flaky:
            ind[expert] = int(not self.has_missing_values(features[expert]))

        # store items in a flat dictionary to simplify aggregation
        sample = {"text": self.text_features["train"][idx]}
        sample.update({f"{key}_ind": val for key, val in ind.items()})
        sample.update({f"{key}_sz": val[idx] for key, val in
                       self.expert_feat_sizes.items()})
        sample["text_sz"] = self.text_feature_sizes["train"][idx]
        sample.update(features)
        return sample

    def collate_data(self, data):

        ind = {expert: [x[f"{expert}_ind"] for x in data]
               for expert in self.ordered_experts}
        ind = {key: ensure_tensor(np.array(val)) for key, val in ind.items()}

        experts = []
        for expert in self.ordered_experts:
            if expert in {"audio", "ocr"}:
                # zero pad the variable length inputs to the shortest possible length
                pad_to = max([x[f"{expert}_sz"] for x in data])
                val = th.from_numpy(np.stack([x[expert][:pad_to] for x in data], axis=0))
            else:
                val = th.from_numpy(np.vstack([x[expert] for x in data]))
            experts.append((expert, val.float()))
        experts = OrderedDict(experts)

        # Similarly, we zero pad the text to the shortest possible length
        pad_to = max([x[f"text_sz"] for x in data])
        text = th.from_numpy(np.array([x["text"][:pad_to] for x in data])).float()
        text = self.unsqueeze_text(text)

        return {"text": text, "experts": experts, "ind": ind}

    def unsqueeze_text(self, text):
        # To enable compatibility with datasets that use multiple
        # captions, we insert a singleton on the second dimension (which is used
        # to denote the number of captions per video)
        if self.captions_per_video == 1:
            return th.unsqueeze(text, 1)
        else:
            raise ValueError("Currently only 1 caption per vid is supported")

    def __len__(self):
        return len(self.text_features["train"])

    def get_retrieval_data(self):
        experts = OrderedDict(
            (expert, th.from_numpy(self.features["val"][expert]).float())
            for expert in self.ordered_experts
        )
        text = self.unsqueeze_text(self.text_retrieval)
        retrieval_data = {"text": text, "experts": experts, "ind": self.test_ind}
        meta = {
            "query_masks": None,
            "raw_captions": self.raw_captions_retrieval,
            "paths": self.video_path_retrieval,
        }
        return retrieval_data, meta

    def zero_pad_to_tensor(self, feats, max_length):
        """Zero-pad a list of variable length features into a single tensor.

        Args:
            feats (list[np.ndarray]): a list of numpy array of features
            max_length (int): the maximum allowed length for any individual feature

        Returns:
            (torch.Tensor, list[int]) the aggregated tensor and a list of the sizes
                 (after cropping to max_length, but before zero padding).
        """
        cropped_sizes = [min(x.shape[0], max_length) for x in feats]
        pad_to = max(cropped_sizes)
        tensor = np.zeros((len(feats), pad_to, feats[0].shape[-1]))
        for ii, (feat_sz, feat_val) in enumerate(zip(cropped_sizes, feats)):
            tensor[ii, 0:feat_sz, :] = feat_val[:feat_sz, :]
        return tensor, cropped_sizes

    def has_missing_values(self, x):
        # We check the first value to look for a missing feature marker (checking
        # the whole array would slow things down and isn't necessary)
        if x.ndim == 1:
            res = np.isnan(x[0])
        elif x.ndim == 2:
            res = np.isnan(x[0][0])
        else:
            raise ValueError("did not expect tensor")
        return res

    def canonical_features(self, feats, keep_zeros=False):
        """Precomputed features should have the same format prior to aggregation.  The
        first dimension is temporal, the second dimension is the feature-dim, unless
        VLAD aggregation is to be used across instances, in which case the second dim
        is the instance dim and the third dim is the feature dim.

        For certain kinds of features, e.g. audio, the zero features must be kept in
        place - this is enabled by `keep_zeros`.
        """
        feats = feats.astype(np.float32)
        if feats.ndim == 2:
            only_zeros = np.sum(feats, axis=1) == 0
            print("found {} missing features".format(only_zeros.sum()))
            if not keep_zeros:
                feats[only_zeros, :] = self.MISSING_VAL
        elif feats.ndim == 3:
            only_zeros = np.sum(np.sum(feats, axis=1), axis=1) == 0
            if not keep_zeros:
                feats[only_zeros, :] = self.MISSING_VAL
        return feats