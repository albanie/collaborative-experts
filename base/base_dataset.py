import time
import inspect
import logging
import json
import functools
from abc import abstractmethod
from typing import Dict, Union, List
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch as th
from numpy.random import randint
from torch.utils.data import Dataset
from typeguard import typechecked

import data_loader
from utils.util import ensure_tensor, expert_tensor_storage
from zsvision.zs_utils import memcache
import pickle

# For SLURM usage, buffering makes it difficult to see events as they happen, so we set
# the global print statement to enforce flushing
print = functools.partial(print, flush=True)


class BaseDataset(Dataset):

    @staticmethod
    @abstractmethod
    @typechecked
    def dataset_paths(training_file=None) -> Dict[str, Union[Path, str]]:
        """Generates a datastructure containing all the paths required to load features
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

    @typechecked
    def __init__(
            self,
            data_dir: Path,
            fuse_captions: bool,
            spatial_feats: bool,
            challenge_mode: bool,
            eval_only: bool,
            use_zeros_for_missing: bool,
            task: str,
            text_agg: str,
            text_feat: str,
            split_name: str,
            cls_partition: str,
            root_feat_folder: str,
            challenge_test_root_feat_folder: str,
            text_dim: int,
            num_test_captions: int,
            restrict_train_captions: int,
            max_tokens: Dict[str, int],
            text_dropout: float,
            logger: logging.Logger,
            raw_input_dims: Dict[str, int],
            feat_aggregation: Dict[str, Dict],
            distil_params: Union[None, Dict],
            training_file: Union[None, str],
            caption_masks: Union[None, str],
            ce_shared_dim: Union[None, int],
            **kwargs,
    ):
        self.task = task
        self.eval_only = eval_only
        self.logger = logger
        self.challenge_mode = challenge_mode
        self.text_feat = text_feat
        self.data_dir = data_dir
        self.text_dim = text_dim
        self.spatial_feats = spatial_feats
        self.text_dropout = text_dropout
        self.restrict_train_captions = restrict_train_captions
        self.max_tokens = max_tokens
        self.cls_partition = cls_partition
        self.fuse_captions = fuse_captions
        self.num_test_captions = num_test_captions
        self.feat_aggregation = feat_aggregation
        self.root_feat = data_dir / root_feat_folder
        self.challenge_test_root_feat_folder = data_dir / challenge_test_root_feat_folder
        self.experts = set(raw_input_dims.keys())
        self.distil_params = distil_params
        self.training_file = training_file

        # This attributes can be overloaded by different datasets, so it must be set
        # before the `load_features() method call`
        self.restrict_test_captions = None
        self.text_features = None
        self.label_features = None
        self.video_labels = None
        self.distil_features = None
        self.raw_captions = None
        self.features = None

        # Use a single caption per video when forming training minibatches (different
        # captions from the same video may still be used across different minibatches)
        self.captions_per_video = 1

        # TODO(Samuel) - is a global fixed ordering still necessary?
        self.ordered_experts = list(raw_input_dims.keys())

        # Training and test lists are set by dataset-specific subclasses
        self.partition_lists = {}
        self.configure_train_test_splits(split_name=split_name)

        # All retrieval-based tasks use a single dataloader (and handle the retrieval
        # data separately), whereas for classification we use one dataloader for
        # training and one for validation.
        self.logger.info("The current task is {}".format(self.task))
        self.sample_list = self.partition_lists["train"]

        self.num_samples = len(self.sample_list)
        num_val = len(self.partition_lists["val"])

        self.ce_shared_dim = ce_shared_dim
        if caption_masks is not None:
            self.caption_masks = pickle.load(open(self.root_feat / Path(caption_masks), "rb"))
        else:
            self.caption_masks = None

        if self.task == "classification":
            self.sample_list = self.partition_lists[self.cls_partition]
            self.num_samples = len(self.sample_list)
            self.logger.info("The current cls_partition is {}".format(self.cls_partition))

            # The number of classes and class type (i.e. single or multi-label) must be
            # overriden in the subclass
            self.num_classes = None
            self.class_type = None

        self.raw_input_dims = raw_input_dims

        # we store default paths to enable visualisations (this can be overloaded by
        # dataset-specific classes)
        self.video_path_retrieval = [f"videos/{x}.mp4" for x
                                     in self.partition_lists["val"]]

        # NOTE: We use nans rather than zeros to indicate missing faces, unless we wish
        # to test single modality strength, which requires passing zeroed features for
        # missing videos
        if use_zeros_for_missing:
            self.MISSING_VAL = 0
        else:
            self.MISSING_VAL = np.nan

        # load the dataset-specific features into memory
        self.load_features()

        if text_agg == "avg":
            self.logger.info("averaging the text features...")
            for key, val in self.text_features.items():
                self.text_features[key] = [np.mean(x, 0, keepdims=1) for x in val]
            self.logger.info("finished averaging the text features")

        self.trn_config = {}
        self.raw_config = {}
        self.tensor_storage = expert_tensor_storage(self.experts, self.feat_aggregation)
        for static_expert in self.tensor_storage["fixed"]:
            if static_expert in self.feat_aggregation:
                if "trn_seg" in self.feat_aggregation[static_expert].keys():
                    self.trn_config[static_expert] = \
                        self.feat_aggregation[static_expert]["trn_seg"]
                if "raw" in self.feat_aggregation[static_expert]["temporal"]:
                    self.raw_config[static_expert] = 1

        if self.task == "classification":
            # for classification we don't need to preload additional features
            return

        retrieval = {
            expert: np.zeros((num_val, self.max_tokens[expert], raw_input_dims[expert]))
            for expert in self.tensor_storage["variable"]
        }

        retrieval.update({expert: np.zeros((num_val, raw_input_dims[expert]))
                          for expert in self.tensor_storage["fixed"]})
        self.retrieval = retrieval
        self.test_ind = {expert: th.ones(num_val) for expert in self.experts}
        self.raw_captions_retrieval = [None] * num_val

        if self.task == "retrieval-as-classification":
            # Treat each category label as a query
            num_labels = len(self.label_features)
            self.text_retrieval = np.zeros((num_labels, 1, 1, self.text_dim))
            self.query_masks = np.zeros((num_labels, num_val))
            for ii, video_name in enumerate(self.partition_lists["val"]):
                labels = self.video_labels[video_name]
                self.query_masks[np.array(labels), ii] = 1

            # Perform a single loop over the categories and encode the average label
            # as queries
            for ii, embedding in self.label_features.items():
                self.text_retrieval[ii, :, :, :] = np.mean(embedding, axis=0, keepdims=1)

        elif self.task == "retrieval":
            # avoid evaluation on missing queries
            self.query_masks = np.zeros((num_val, num_test_captions))
            self.text_token_mask = np.zeros((num_val, num_test_captions))
            self.text_retrieval = np.zeros((num_val, self.num_test_captions,
                                            self.max_tokens["text"], self.text_dim))
        else:
            raise ValueError(f"Unrecognised task: {self.task}")

        for ii, video_name in enumerate(self.partition_lists["val"]):

            self.raw_captions_retrieval[ii] = self.raw_captions[video_name]
            for expert in self.tensor_storage["fixed"].intersection(self.experts):
                feats = self.features[expert][video_name]
                drop = self.has_missing_values(feats)

                self.test_ind[expert][ii] = not drop
                self.retrieval[expert][ii] = feats
                if drop:
                    self.retrieval[expert][ii][:] = self.MISSING_VAL
                if self.feat_aggregation[expert].get("binarise", False):
                    keep = np.logical_not(np.isnan(self.retrieval[expert][:, 0, 0]))
                    marker = np.ones_like(self.retrieval[expert][keep])
                    self.retrieval[expert][keep] = marker

            for expert in self.tensor_storage["variable"].intersection(self.experts):
                feats = self.features[expert][video_name]
                drop = self.has_missing_values(feats)
                self.test_ind[expert][ii] = not drop
                if drop:
                    self.retrieval[expert][ii][:] = self.MISSING_VAL
                if self.feat_aggregation[expert].get("binarise", False):
                    keep = np.logical_not(np.isnan(self.retrieval[expert][:, 0, 0]))
                    marker = np.ones_like(self.retrieval[expert][keep])
                    self.retrieval[expert][keep] = marker
                if self.test_ind[expert][ii]:
                    keep = min(self.max_tokens[expert], len(feats))
                    self.retrieval[expert][ii, :keep, :] = feats[:keep]

            if self.task == "retrieval":
                candidates_sentences = self.text_features[video_name]

                if self.restrict_test_captions is not None:
                    keep_sent_idx = self.restrict_test_captions[video_name]
                    candidates_sentences = [candidates_sentences[keep_sent_idx]]

                self.query_masks[ii, :len(candidates_sentences)] = 1

                if self.fuse_captions:
                    # fuse into a single caption
                    text_feats = np.vstack(candidates_sentences)
                    keep = min(len(text_feats), self.max_tokens["text"])
                    self.text_retrieval[ii, 0, :keep, :] = text_feats[:keep, :]
                    self.text_token_mask[ii, 0] = keep
                    self.query_masks[ii, :] = 1
                else:
                    for test_caption_idx in range(self.num_test_captions):
                        if len(candidates_sentences) <= test_caption_idx:
                            break
                        keep = min(len(candidates_sentences[test_caption_idx]),
                                   self.max_tokens["text"])
                        self.text_token_mask[ii, test_caption_idx] = keep
                        if ii % 500 == 0 and test_caption_idx == 0:
                            msg = (
                                f"{ii}/{len(self.partition_lists['val'])} will evaluate "
                                f"sentence {test_caption_idx} out of "
                                f"{len(candidates_sentences)} (has {keep} words) "
                                f"{video_name}"
                            )
                            self.logger.info(msg)
                        text_feats = candidates_sentences[test_caption_idx][: keep]
                        if text_feats.size == 0:
                            text_feats = 0
                            raise ValueError("empty text features!")
                        self.text_retrieval[ii, test_caption_idx, :keep, :] = text_feats

        self.sanity_checks()

    def configure_train_test_splits(self, split_name):
        """Partition the datset into train/val/test splits.

        Args:
            split_name (str): the name of the split
        """
        self.paths = type(self).dataset_paths(self.training_file)
        print("loading training/val splits....")
        tic = time.time()
        for subset, path in self.paths["subset_list_paths"][split_name].items():
            if self.challenge_mode and split_name == "public_server_test" \
                    and subset == "val":
                root_feat = Path(self.challenge_test_root_feat_folder)
            else:
                root_feat = Path(self.root_feat)
            subset_list_path = root_feat / path
            if subset == "train" and self.eval_only:
                rows = []
            else:
                with open(subset_list_path) as f:
                    rows = f.read().splitlines()
                if isinstance(self, data_loader.DiDeMo_dataset.DiDeMo):
                    # For DiDeMo, we need to remove additional video suffixes
                    rows = [x.strip().split(".")[0] for x in rows]
            self.partition_lists[subset] = rows
        print("done in {:.3f}s".format(time.time() - tic))
        self.split_name = split_name

    def collate_data(self, data):
        batch_size = len(data)
        tensors = {}
        for expert in self.tensor_storage["fixed"]:
            if expert in self.trn_config.keys():
                tensors[expert] = np.zeros((batch_size, self.trn_config[expert],
                                            self.raw_input_dims[expert]))
            else:
                tensors[expert] = np.zeros((batch_size, self.raw_input_dims[expert]))
        if self.distil_features is not None:
            distil = {}
            distil_text = {}

            for t in self.distil_features:
                distil[t] = {}
                distil_text[t] = {}

                check_moee = False
                if self.distil_params is not None and 'moee' in self.distil_params:
                    if isinstance(self.distil_params['moee'], list) and self.distil_params['moee'][t] == 1:
                        check_moee = True
                    elif not isinstance(self.distil_params['moee'], list) and self.distil_params['moee'] == True:
                        check_moee = True
                for mod in self.distil_features[t][list(self.distil_features[t].keys())[0]]['vid_embds'].keys():
                    if check_moee:
                        size = self.raw_input_dims[mod]
                        distil[t][mod] = np.zeros((batch_size, size))
                        distil_text[t][mod] = np.zeros((batch_size, 1, size))
                    else:
                        distil[t][mod] = np.zeros((batch_size, self.ce_shared_dim))
                        distil_text[t][mod] = np.zeros((batch_size, 1, self.ce_shared_dim))

        # Track which indices of each modality are available in the present batch
        ind = {expert: np.zeros(batch_size) for expert in self.experts}
        tensors.update({expert: np.zeros(
            (batch_size, self.max_tokens[expert], self.raw_input_dims[expert])
        ) for expert in self.tensor_storage["variable"]})

        if "retrieval" in self.task:
            text_tensor = np.zeros((batch_size, self.captions_per_video,
                                    self.max_tokens["text"], self.text_dim))
            text_token_mask = np.zeros((batch_size, self.captions_per_video))
        elif "classification" in self.task and self.class_type == "single_label":
            label_tensor = np.zeros(batch_size)
            vid_name = []
        elif "classification" in self.task and self.class_type == "multi_label":
            label_tensor = np.zeros((batch_size, self.num_classes))
            vid_name = []

        for ii, _ in enumerate(data):
            datum = data[ii]
            for expert in self.experts:
                ind[expert][ii] = datum[f"{expert}_ind"]
            for expert in self.tensor_storage["fixed"]:
                tensors[expert][ii] = datum[expert]
            for expert in self.tensor_storage["variable"]:
                if ind[expert][ii]:
                    keep = min(len(datum[expert]), self.max_tokens[expert])
                    if keep:
                        tensors[expert][ii, :keep, :] = datum[expert][:keep]
                else:
                    tensors[expert][ii, :, :] = self.MISSING_VAL

            if self.distil_features is not None:
                for t in datum['distil_mods']:
                    for mod in datum['distil_mods'][t]:
                        distil[t][mod][ii] = datum['distil_mods'][t][mod]
                for t in datum['distil_mods']:
                    for mod in datum['distil_texts'][t]:
                        distil_text[t][mod][ii] = datum['distil_texts'][t][mod]

            if "retrieval" in self.task:
                text = datum["text"]
                for jj in range(self.captions_per_video):
                    keep = min(len(text[jj]), self.max_tokens["text"])
                    text_tensor[ii, jj, :keep, :] = text[jj][:keep]
                    text_token_mask[ii, jj] = keep
            elif self.task == "classification":
                if self.cls_partition != 'test':
                    label_tensor[ii] = datum["labels"]
                vid_name.append(datum["vid"])

        ind = {key: ensure_tensor(val) for key, val in ind.items()}
        experts = OrderedDict(
            (expert, th.from_numpy(tensors[expert]).float())
            for expert in self.ordered_experts)
        if self.distil_features is not None:
            for t in distil:
                distil[t] = OrderedDict(
                    (expert, th.from_numpy(distil[t][expert]).float())
                    for expert in distil[t])
                distil_text[t] = OrderedDict(
                    (expert, th.from_numpy(distil_text[t][expert]).float())
                    for expert in distil_text[t])

        for expert in self.experts:
            if self.feat_aggregation[expert].get("binarise", False):
                replace = np.logical_not(th.isnan(experts[expert][:, 0, 0]))
                experts[expert][replace] = th.ones_like(experts[expert][replace])

        minibatch = {"experts": experts, "ind": ind}
        if self.distil_features is not None:
            minibatch["distil_video"] = distil
            minibatch["distil_text"] = distil_text
        if "retrieval" in self.task:
            minibatch["text"] = th.from_numpy(text_tensor).float()
            minibatch["text_token_mask"] = th.from_numpy(text_token_mask)

        elif self.task == "classification":
            if self.cls_partition != 'test':
                minibatch["labels"] = th.from_numpy(label_tensor).float()
            if self.cls_partition != "train":
                # we only pass the video names for visualisation and making predictions
                # on the val/test set
                minibatch["vid_name"] = vid_name
        return minibatch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < self.num_samples:
            vid = self.sample_list[idx]
            if self.caption_masks is not None:
                caption_masks = self.caption_masks[vid]
            else:
                caption_masks = None
            if self.distil_features is not None:
                distil_mod_feats = {}
                distil_text_feats = {}

                for t in self.distil_features:
                    distil_mod_feats[t] = {}
                    distil_text_feats[t] = {}
                    for k in self.distil_features[t][vid]:
                        for mod in self.distil_features[t][vid][k]:
                            if k == 'vid_embds':
                                distil_mod_feats[t][mod] = self.distil_features[t][vid][k][mod]
                            elif k == 'text_embds':
                                distil_text_feats[t][mod] = self.distil_features[t][vid][k][mod]

            # try:
            features = {}
            for expert in self.experts:
                if expert not in self.trn_config.keys():
                    try:
                        if expert in self.raw_config.keys():
                            features[expert] = np.mean(self.features[expert][vid], axis=0)
                        else:
                            features[expert] = self.features[expert][vid]
                    except:
                        import ipdb; ipdb.set_trace()
                else:
                    # ------------------------------------
                    # Yang's implementation for TRN inputs
                    # ------------------------------------
                    raw_frame_feats = self.features[expert][vid]
                    new_length = 1
                    num_frames = raw_frame_feats.shape[0]
                    avg_duration = ((num_frames - new_length + 1)
                                    // self.trn_config[expert])
                    assert avg_duration > 0, "average duration must be positive"
                    if avg_duration > 0:
                        # maybe we could change to use average for each tiny segment
                        # seems like use everything per iter
                        offsets = np.multiply(list(range(self.trn_config[expert])),
                                              avg_duration)
                        offsets += randint(avg_duration, size=self.trn_config[expert])
                        new_frame_feats = np.zeros((self.trn_config[expert],
                                                    raw_frame_feats.shape[1]))
                        for idx, xx in enumerate(offsets):
                            # yang! you might want to change back
                            new_frame_feats[idx, :] = raw_frame_feats[xx, :]
                        msg = "returning a wrong feature != segment num"
                        assert new_frame_feats.shape[0] == self.trn_config[expert], msg
                        features[expert] = new_frame_feats

            ind = {}
            for expert in self.ordered_experts:
                if expert in self.tensor_storage["flaky"]:
                    ind[expert] = not self.has_missing_values(features[expert])
                else:
                    ind[expert] = 1

            if self.task in {"retrieval", "retrieval-as-classification"}:
                # Handle some inconsistencies between how the text features are stored
                text = self.text_features[vid]
                if self.fuse_captions:
                    text = [np.vstack(text)]
                    pick = None
                elif isinstance(text, list):
                    if caption_masks is not None:
                        probability = caption_masks / np.sum(caption_masks)
                        pick = np.random.choice(len(text), size=self.captions_per_video, p=probability)
                        assert caption_masks[pick] == 1
                    else:
                        pick = np.random.choice(len(text), size=self.captions_per_video)
 
                    text = np.array(text)[pick]
                    if self.distil_features is not None:
                        # Select appropriate text embd for teacher data
                        for t in distil_text_feats:
                            for mod in distil_text_feats[t]:
                                distil_text_feats[t][mod] = np.array(distil_text_feats[t][mod])[pick]
                else:
                    pick = None
                    text = np.random.choice(text, size=self.captions_per_video)

                if np.random.random() < self.text_dropout:
                    if pick is not None:
                        mask = np.random.random(len(text[0]))
                        text = [text[0][mask > 0.5]]
                    else:
                        raise NotImplementedError("TODO: Add dropouot for picked text")

        # Return both the missing indices as well as the tensors
        if self.task in {"retrieval", "retrieval-as-classification"}:
            if self.distil_features is not None:
                sample = {"text": text, "distil_mods": distil_mod_feats, "distil_texts": distil_text_feats}
            else:
                sample = {"text": text}
        elif self.task == "classification":
            if self.class_type == "single_label":
                labels = self.video_labels[vid]
                assert len(labels) == 1, "expected single label"
                labels = labels[0]
            elif self.class_type == "multi_label":
                if self.cls_partition != 'test':
                    labels = np.zeros(self.num_classes)
                    labels[self.video_labels[vid]] = 1
            else:
                raise ValueError(f"unknown label class type: {self.class_type}")
            sample = {}
            if self.cls_partition != 'test':
                sample = {"labels": labels}
            sample.update({"vid": vid})
        else:
            raise ValueError(f"unknown task: {self.task}")
        sample.update({f"{key}_ind": val for key, val in ind.items()})
        sample.update(features)
        return sample

    def get_retrieval_data(self):
        experts = OrderedDict(
            (expert, th.from_numpy(self.retrieval[expert]).float())
            for expert in self.ordered_experts
        )
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
        return retrieval_data, meta

    def has_missing_values(self, x):
        return isinstance(x, float) and np.isnan(x)

    def visual_feat_paths(self, model_spec, tag=None):
        """Canonical path lookup for visual features
        """
        if model_spec not in self.ordered_experts:
            self.logger.info(f"Skipping load for {model_spec} (feature not requested)")
            return f"SKIPPED-{model_spec}"

        feat_type, model_name, _ = model_spec.split(".")
        aggs = self.feat_aggregation[model_spec]
        base = f"aggregated_{feat_type.replace('-', '_')}"
        required = ("fps", "pixel_dim", "stride")
        fps, pixel_dim, stride = [aggs.get(x, None) for x in required]
        if feat_type in {"facecrops", "faceboxes"}:
            base = f"{base}_{fps}fps_{pixel_dim}px_stride{stride}"
        elif feat_type not in {"ocr", "speech", "audio"}:
            base = f"{base}_{fps}fps_{pixel_dim}px_stride{stride}"

        for option in "offset", "inner_stride", "num_segments":
            if aggs.get(option, None) is not None:
                base += f"_{option}{aggs[option]}"

        feat_paths = []
        for agg in aggs["temporal"].split("-"):
            fname = f"{model_name}-{agg}"
            if aggs["type"] == "logits":
                fname = f"{fname}-logits"
            if tag is not None:
                fname += f"-{tag}"
            feat_paths.append(Path(base) / f"{fname}.pickle")
        return feat_paths

    def log_assert(self, bool_, msg="", verbose=True):
        """Use assertions that will be written to the logs. This is a recipe from:
        http://code.activestate.com/recipes/577074-logging-asserts/
        """
        try:
            assert bool_, msg
        except AssertionError:
            # construct an exception message from the code of the calling frame
            last_stackframe = inspect.stack()[-2]
            source_file, line_no, func = last_stackframe[1:4]
            source = f"Traceback (most recent call last):\n" + \
                     f" File {source_file}, line {line_no}, in {func}\n"
            if verbose:
                # include more lines than that where the statement was made
                source_code = open(source_file).readlines()
                source += "".join(source_code[line_no - 3:line_no + 1])
            else:
                source += last_stackframe[-2][0].strip()
            self.logger.debug(f"{msg}\n{source}")
            raise AssertionError(f"{msg}\n{source}")

    def summary_stats(self):
        """Report basic statistics about feature availability and variable lengths
        across the different subsets of the data.
        """
        self.logger.info("Computing feature stats...")
        queries = self.ordered_experts + ["text"]
        for subset, keep in self.partition_lists.items():
            keep = set(keep)
            print(f"Summary for {subset}")
            for expert in queries:
                if expert in self.features:
                    feats = self.features[expert]
                else:
                    feats = self.text_features
                vals = [feats[key] for key in keep]
                missing = 0
                sizes = []
                for val in vals:
                    if self.has_missing_values(val):
                        missing += 1
                    else:
                        sizes.append(len(val))
                if sizes:
                    stat_str = (f"min: {np.min(sizes):4}, "
                                f"max: {np.max(sizes):4}, "
                                f"mean: {np.mean(sizes):.1f}")
                    print(f"{subset}: missing: {missing:4}, {stat_str} {expert}")

    @staticmethod
    @typechecked
    def common_text_feat_paths() -> Dict[str, str]:
        """Load the text features and raw captions to be used in the challenge.
        """
        with open("model/text_embedding_models.json") as f:
            supported_text_embeddings = json.load(f)
        return {name: f"{name}.pkl" for name in supported_text_embeddings}

    @staticmethod
    @typechecked
    def common_feat_names() -> List[str]:
        """Produce a common collection of feature names shared amongst datasets.
        """
        feature_names = [
            "imagenet.senet154.0",
            "scene.densenet161.0",
            "imagenet.resnext101_32x48d.0",
            "trn.moments-trn.0",
            "moments_2d.resnet50.0",
            "i3d.i3d.0",
            "i3d.i3d.1",
            "s3dg.s3dg.0",
            "s3dg.s3dg.1",
            "r2p1d.r2p1d-ig65m.0",
            "r2p1d.r2p1d-ig65m.1",
            "r2p1d.r2p1d-ig65m-kinetics.0",
            "r2p1d.r2p1d-ig65m-kinetics.1",
            "moments_3d.moments-resnet3d50.0",
            "moments_3d.moments-resnet3d50.1"
        ]
        return feature_names

    def load_challenge_text_features(self):
        """Load the text features and raw captions to be used in the challenge.
        """
        text_feat_path = self.paths["challenge_text_feat_paths"][self.text_feat]
        if self.split_name == "public_server_test":
            text_path = self.challenge_test_root_feat_folder / text_feat_path
            caption_path = (self.challenge_test_root_feat_folder /
                            self.paths["raw_captions_path"])
        else:
            text_path = Path(self.root_feat) / text_feat_path
            caption_path = Path(self.root_feat) / self.paths["raw_captions_path"]
        self.text_features = memcache(text_path)
        self.raw_captions = memcache(caption_path)
