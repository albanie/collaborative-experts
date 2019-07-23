"""

NOTE: This code is loosely based on the LSMDC dataset loader provided in the MoEE
codebase:
https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/LSMDC.py
"""

import torch as th
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from utils.util import memcache
from pathlib import Path


class LSMDC(Dataset):

    def __init__(self, data_dir, feat_aggregation, raw_input_dims, num_test_captions,
                 split_name, text_dim, text_feat, rgb_model_name, fuse_captions,
                 max_text_words, max_expert_tokens, verbose=False):

        self.ordered_experts = list(raw_input_dims.keys())
        self.captions_per_video = 1
        self.MISSING_VAL = np.nan
        self.text_feat = text_feat
        # self.text_features = text_features
        self.max_text_words = max_text_words
        self.raw_input_dims = raw_input_dims
        root_feat = Path(data_dir) / "symlinked-feats"

        print("Reading test data ...")
        train_feat_names = {
            "face": "X_face.npy",
            "flow": "X_flow.npy",
            "rgb": "X_resnet.npy",
            "scene": "densenet161-scene-avg-train.npy",
            "ocr": "w2v-ocr-raw-train.npy",
            "audio": "X_audio_train.npy",
        }
        val_feat_names = {
            "face": "face-retrieval.npy.tensor.npy",
            "flow": "flow-retrieval.npy.tensor.npy",
            "rgb": "resnet152-retrieval.npy.tensor.npy",
            "scene": "densenet161-scene-avg-val.npy",
            "ocr": "w2v-ocr-raw-val.npy",
            "audio": "X_audio_retrieval.npy.tensor.npy",
        }
        feat_paths = {"train": train_feat_names, "val": val_feat_names}
        text_paths = {"train": "w2v_LSMDC.npy", "val": "w2v_LSMDC_retrieval.npy"}
        # feat_paths = {key: Path(root_feat) / value for key, value in feat_names.items()}

        features = {}
        for key, feat_names in feat_paths.items():
            features[key] = {expert: memcache(Path(root_feat) / path) for expert, path
                             in feat_names.items()}
        text_features = {key: self.text2fixed_length(memcache(Path(root_feat) / val))
                         for key, val in text_paths.items()}

        with open(Path(root_feat) / "test_video_paths.txt", "r") as f:
            self.video_path_retrieval = [Path(x) for x in f.read().splitlines()]

        # store the indices of missing face and ocr features, marking the other experts
        # as available
        self.flaky = {"face", "ocr"}
        ind_paths = {Path(root_feat) / f"no_{x}_ind_retrieval.npy" for x in self.flaky}
        test_ind = {expert: 1 - memcache(path) for expert, path in ind_paths.items()}
        test_ind.update({expert: np.ones_like(test_ind["ocr"]) for expert in features
                         if expert not in self.flaky})
        missing = np.sum(features["val"]["face"], axis=1) == 0
        features["val"]["face"][missing, :] = np.nan
        missing = np.sum(np.sum(features["val"]["ocr"], axis=1), axis=1) == 0
        features["val"]["ocr"][missing, :] = np.nan
        self.features = features
        self.text_retrieval = self.make_text_tensor(text_features["val"])

    def get_retrieval_data(self):

        # retrieval_vals = {
        #     "video": th.from_numpy(vid_retrieval).float(),
        #     "flow": th.from_numpy(flow_retrieval).float(),
        #     "face": th.from_numpy(face_retrieval).float(),
        #     "audio": th.from_numpy(audio_retrieval).float(),
        #     "scene": th.from_numpy(scene_retrieval).float(),
        #     "ocr": th.from_numpy(ocr_retrieval).float(),
        #     "paths": absolute_video_paths,
        #     "raw_captions": raw_captions["val"],
        #     "text": self.make_text_tensor(text_retrieval),
        #     "face_ind": 1 - face_ind_test,
        #     "ocr_ind": 1 - ocr_ind_test,
        # }
        # dataloaders["retrieval"] = retrieval_vals
        # self.text_sizes = np.array(list(map(len, text_features))).astype(int)
        # self.video_features_size = raw_input_dims["visual"]
        # self.audio_features_size = raw_input_dims["audio"]
        # self.flow_features_size = raw_input_dims["motion"]
        # self.face_features_size = raw_input_dims["face"]
        # self.ocr_features_size = raw_input_dims["ocr"]

        experts = OrderedDict(
            (expert, th.from_numpy(self.features["val"][expert]).float())
            for expert in self.ordered_experts
        )
        retrieval_data = {
            "text": self.text_retrieval,
            "experts": experts,
            "ind": self.test_ind,
        }
        meta = {
            "query_masks": None,
            "raw_captions": self.raw_captions_retrieval,
            "paths": self.video_path_retrieval,
        }
        return retrieval_data, meta

    def __len__(self):
        return len(self.text_features["train"])

    def __getitem__(self, idx):

        # flow = self.flow_features[idx]
        # face = self.face_features[idx]
        # ocr = self.ocr_features[idx]
        # audio = self.audio_features[idx]
        # audio_size = self.audio_sizes[idx]
        features = {expert: self.features[expert][idx] for expert in self.ordered_experts}
        ind = {expert: 1 for expert in self.ordered_experts}
        for expert in self.flaky:
            ind[expert] = not self.has_missing_values(features[expert])
        sample = {"text": self.text_features["train"][idx]}
        sample.update({f"{key}_ind": val for key, val in ind.items()})
        sample.update(features)
        return sample

    def collate_data(self, data):
        import ipdb; ipdb.set_trace()

        # return {
        #     "video": self.visual_features[idx],
        #     "scene": self.scene_features[idx],
        #     "flow": flow,
        #     "face": face,
        #     "face_ind": face_ind,
        #     "ocr": ocr,
        #     "ocr_ind": ocr_ind,
        #     "text": self.text_features[idx],
        #     "audio": audio,
        #     "audio_size": audio_size,
        #     "text_size": self.text_sizes[idx],
        # }

    def make_text_tensor(self, xlist):
        max_len = max(map(len, xlist))
        assert self.captions_per_video == 1, "additional captions not supported"
        caption_idx = 0
        tensor = np.zeros((len(xlist), self.captions_per_video, max_len, xlist[0].shape[-1]))
        for ii in range(len(xlist)):
            if len(xlist[ii]):
                keep = min(max_len, xlist[ii].shape[0])
                cropped = xlist[ii][:keep]
                tensor[ii, caption_idx, :keep, :] = cropped
        return th.from_numpy(tensor).float()

    def shorteningTextTensor(self, text_features, text_sizes):
        return text_features[:, 0:int(max(text_sizes)), :]

    def text2fixed_length(self, text_features):
        text_sizes = list(map(len, text_features))
        text_tensors = np.zeros((len(text_features), self.captions_per_video,
                                 self.max_text_words, self.text_features[0].shape[1]))
        assert self.captions_per_video == 1, "expected a single caption per video"
        caption_idx = 0
        print("restricting features to fixed length")
        for jj in range(len(text_features)):
            text_feats_ = text_features[jj]
            sz_ = text_sizes[jj]
            if sz_ > self.max_text_words:
                text_tensors[jj, caption_idx] = text_feats_[0:self.max_text_words, :]
            else:
                text_tensors[jj, caption_idx, 0:sz_, :] = text_feats_
        return text_tensors

    def has_missing_values(self, x):
        # We check the first value to look for a missing feature marker (checking
        # the whole array would slow things down and isn't necessary)
        if len(x.size()) == 1:  # "expected to check vector"
            res = th.isnan(x[0]).item()  # NOTE: we cannot check equality against a NaN
        elif len(x.size()) == 2:
            res = th.isnan(x[0][0]).item()
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


# class LSMDC(LSMDC_Common):
#     """LSMDC dataset."""

#     def __init__(self, data_dir, feat_aggregation, raw_input_dims, num_test_captions,
#                  split_name, text_dim, text_feat, rgb_model_name, fuse_captions,
#                  max_text_words, max_expert_tokens, verbose=False):
#     # def __init__(
#     #     self,
#     #     visual_features,
#     #     text_features,
#     #     audio_features,
#     #     flow_features,
#     #     face_features,
#     #     ocr_features,
#     #     scene_features,
#     #     struct_constraint,
#     #     struct_constraint_size,
#     #     raw_input_dims,
#     #     coco_visual_path="../Antoine-pytorch/data/X_train2014_resnet152.npy",
#     #     coco_text_path="../Antoine-pytorch/data/w2v_coco_train2014_1.npy",
#     #     coco=True,
#     #     max_words=30,
#     #     text_dim=300,
#     #     verbose=False,
#     # ):
#         """
#         Args:
#         """
#         super().__init__(max_text_words=max_text_words, raw_input_dims=raw_input_dims)

#         # self.visual_features = np.load(clip_path, encoding="latin1")
#         # self.flow_features = np.load(flow_path, encoding="latin1")
#         # self.face_features = np.load(face_path, encoding="latin1")
#         # self.audio_features = np.load(audio_features, encoding="latin1")
#         # self.text_features = np.load(text_features, encoding="latin1")
#         import ipdb; ipdb.set_trace()

#         self.visual_features = visual_features
#         self.flow_features = flow_features
#         self.face_features = face_features
#         self.ocr_features = ocr_features
#         self.scene_features = scene_features
#         self.audio_features = audio_features
#         self.text_features = text_features

#         audio_sizes = list(map(len, self.audio_features))
#         self.audio_sizes = np.array(audio_sizes)

#         self.text_dim = text_dim


#         mask = self.text_sizes > 0

#         self.text_features = self.text_features[mask]
#         self.text_sizes = self.text_sizes[mask]
#         self.visual_features = self.visual_features[mask]
#         self.flow_features = self.flow_features[mask]
#         self.scene_features = self.scene_features[mask]
#         self.face_features = self.face_features[mask]
#         self.ocr_features = self.ocr_features[mask]
#         self.audio_features = self.audio_features[mask]
#         self.audio_sizes = self.audio_sizes[mask]
#         self.audio_sizes.astype(int)

#         self.max_len_audio = max(self.audio_sizes)

#         audio_tensors = np.zeros(
#             (
#                 len(self.audio_features),
#                 max(self.audio_sizes),
#                 self.audio_features[0].shape[1],
#             )
#         )

#         for j in range(len(self.audio_features)):
#             audio_tensors[j, 0 : self.audio_sizes[j], :] = self.audio_features[j]

#         self.n_lsmdc = len(self.visual_features)
#         self.coco_ind = np.zeros((self.n_lsmdc))

#         text_tensors = self.text2fixed_length(text_sizes=self.text_sizes)

#         self.text_features = th.from_numpy(text_tensors)
#         self.text_features = self.text_features.float()
#         self.audio_features = th.from_numpy(audio_tensors)
#         self.audio_features = self.audio_features.float()
#         self.flow_features = th.from_numpy(self.flow_features)
#         self.flow_features = self.flow_features.float()
#         self.scene_features = th.from_numpy(self.scene_features)
#         self.scene_features = self.scene_features.float()
#         self.visual_features = th.from_numpy(self.visual_features)
#         self.visual_features = self.visual_features.float()

#         face_features = self.canonical_features(self.face_features)
#         self.face_features = th.from_numpy(face_features).float()

#         ocr_features = self.canonical_features(self.ocr_features)
#         self.ocr_features = th.from_numpy(ocr_features).float()
#         print("finished LSMDC init")

#     def __len__(self):
#         return len(self.text_features)

#     def __getitem__(self, idx):

#         face_ind = 1
#         ocr_ind = 1

#         if idx >= self.n_lsmdc:
#             flow = th.zeros(self.flow_features_size)
#             face = th.zeros(self.face_features_size)
#             audio = th.zeros(self.audio_features.size()[1], self.audio_features_size)
#             raise ValueError("unexpected idx")
#             audio_size = 1
#             face_ind = 0
#         else:
#             flow = self.flow_features[idx]
#             face = self.face_features[idx]
#             ocr = self.ocr_features[idx]
#             audio = self.audio_features[idx]
#             audio_size = self.audio_sizes[idx]

#             if self.has_missing_values(face):
#                 face_ind = 0
#             if self.has_missing_values(ocr):
#                 ocr_ind = 0
#         return {
#             "video": self.visual_features[idx],
#             "scene": self.scene_features[idx],
#             "flow": flow,
#             "face": face,
#             "face_ind": face_ind,
#             "ocr": ocr,
#             "ocr_ind": ocr_ind,
#             "text": self.text_features[idx],
#             "audio": audio,
#             "audio_size": audio_size,
#             "text_size": self.text_sizes[idx],
#         }


# class LSMDC_qcm(LSMDC_Common):
#     """LSMDC dataset."""

#     def __init__(
#         self,
#         visual_features,
#         text_features,
#         audio_features,
#         flow_features,
#         face_features,
#         scene_features,
#         ocr_features,
#         struct_constraint,
#         struct_constraint_size,
#         raw_input_dims,
#         max_words=30,
#         text_dim=300,
#     ):
#         """
#         Args:
#         """
#         super().__init__(
#             max_words=max_words,
#             struct_constraint=struct_constraint,
#             raw_input_dims=raw_input_dims,
#             struct_constraint_size=struct_constraint_size,
#         )
#         self.visual_features = visual_features
#         self.flow_features = flow_features
#         self.scene_features = scene_features
#         self.face_features = self.canonical_features(face_features)
#         self.ocr_features = self.canonical_features(ocr_features)
#         self.audio_features = audio_features
#         self.text_features = text_features
#         print("features loaded")

#         audio_sizes = list(map(len, self.audio_features))
#         self.audio_sizes = np.array(audio_sizes)
#         self.text_dim = text_dim
#         text_sizes = list(map(len, self.text_features))
#         self.text_sizes = np.array(text_sizes)
#         self.text_sizes = self.text_sizes.astype(int)

#         self.max_len_audio = max(self.audio_sizes)

#         audio_tensors = np.zeros(
#             (
#                 len(self.audio_features),
#                 max(self.audio_sizes),
#                 self.audio_features[0].shape[1],
#             )
#         )

#         for j in range(len(self.audio_features)):
#             audio_tensors[j, 0 : self.audio_sizes[j], :] = self.audio_features[j]

#         text_tensors = self.text2fixed_length(
#             max_words=max_words,
#             text_sizes=self.text_sizes,
#         )

#         self.text_features = th.from_numpy(text_tensors)
#         self.text_features = self.text_features.float()

#         self.audio_features = th.from_numpy(audio_tensors)
#         self.audio_features = self.audio_features.float()

#         self.flow_features = th.from_numpy(self.flow_features)
#         self.flow_features = self.flow_features.float()

#         self.scene_features = th.from_numpy(self.scene_features)
#         self.scene_features = self.scene_features.float()

#         self.visual_features = th.from_numpy(self.visual_features)
#         self.visual_features = self.visual_features.float()

#         self.face_features = th.from_numpy(self.face_features)
#         self.face_features = self.face_features.float()

#         self.ocr_features = th.from_numpy(self.ocr_features)
#         self.ocr_features = self.ocr_features.float()

#     def __len__(self):
#         return len(self.visual_features)

#     def __getitem__(self, tidx):

#         idx, idx2 = tidx

#         face_ind = 1
#         ocr_ind = 1

#         flow = self.flow_features[idx]
#         scene = self.scene_features[idx]
#         face = self.face_features[idx]
#         ocr = self.ocr_features[idx]
#         audio = self.audio_features[idx]
#         audio_size = self.audio_sizes[idx]
#         visual = self.visual_features[idx]

#         # if th.sum(face) == 0:
#         #     face_ind = 0
#         if self.has_missing_values(face):
#             face_ind = 0
#         if self.has_missing_values(ocr):
#             ocr_ind = 0

#         return {
#             "video": visual,
#             "flow": flow,
#             "scene": scene,
#             "face": face,
#             "ocr": ocr,
#             "text": self.text_features[idx2],
#             "audio": audio,
#             "face_ind": face_ind,
#             "ocr_ind": ocr_ind,
#             "audio_size": audio_size,
#             "text_size": self.text_sizes[idx2],
#         }
