"""Joint Video-Text embedding model - an extension of MoEE to include Collaborative
Experts.

Original licence below from MoEE below.
"""
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections import OrderedDict
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from model.net_vlad import NetVLAD
from base import BaseModel
import ipdb


def drop_nans(x, ind, validate_missing):
    """Remove nans, which we expect to find at missing indices.

    Args:
        x (th.Tensor): features
        ind (th.Tensor): binary values denoting whether or not a given feature is
            present
        validate_missing (bool): whether to validate that the missing location contains
            a nan.

    Returns:
        (th.tensor): the features, with the missing values masked to zero.
    """
    missing = th.nonzero(ind == 0).flatten()
    if missing.numel():
        if validate_missing:
            vals = x[missing[0]]
            if not th.isnan(vals.view(-1)[0]):
                ipdb.set_trace()
            assert vals.view(-1)[0], "expected nans at missing locations"
        # Prevent overwrite of the original tensor
        # x_ = x.clone()
        # TODO(samuel): This doesn't do anything, so can remove it
        x_ = x
        x_[missing] = 0
        if th.isnan(x_).sum():
            ipdb.set_trace()
        x = x_
    return x


class CENet(BaseModel):
    def __init__(self, text_dim, use_ce, l2renorm, vlad_clusters,
                 disable_nan_checks, expert_dims,
                 keep_missing_modalities, test_caption_mode, randomise_feats,
                 freeze_weights=False, verbose=False):
        super().__init__()

        self.expert_dims = expert_dims
        self.l2renorm = l2renorm
        self.freeze_weights = freeze_weights
        self.disable_nan_checks = disable_nan_checks
        self.text_pooling = NetVLAD(
            feature_size=text_dim,
            cluster_size=vlad_clusters["text"],
        )
        if randomise_feats:
            self.random_feats = set([x for x in args.randomise_feats.split(",")])
        else:
            self.random_feats = set()

        # sanity checks
        expected_feat_sizes = {"audio": 128, "speech": 300, "ocr": 300}
        self.pooling = nn.ModuleDict()
        for mod, expected in expected_feat_sizes.items():
            if mod in expert_dims.keys():
                feature_size = expert_dims[mod][0] // vlad_clusters[mod]
                msg = f"expected {expected} for {mod} features atm"
                assert feature_size == expected, msg
                self.pooling[mod] = NetVLAD(
                    feature_size=feature_size,
                    cluster_size=vlad_clusters[mod],
                )

        self.ce = CEModule(
            use_ce=use_ce,
            verbose=verbose,
            l2renorm=l2renorm,
            random_feats=self.random_feats,
            freeze_weights=freeze_weights,
            text_dim=self.text_pooling.out_dim,
            test_caption_mode=test_caption_mode,
            expert_dims=expert_dims,
            disable_nan_checks=disable_nan_checks,
            keep_missing_modalities=keep_missing_modalities,
        )

    def randomise_feats(self, experts, key):
        if key in self.random_feats:
            # keep expected nans
            nan_mask = th.isnan(experts[key])
            experts[key] = th.randn_like(experts[key])
            if not self.disable_nan_checks:
                nans = th.tensor(float('nan'))  # pylint: disable=not-callable
                experts[key][nan_mask] = nans.to(experts[key].device)
        return experts

    def forward(self, text, experts, ind, raw_captions=None):
        aggregated_experts = OrderedDict()

        for mod in ("face", "flow", "scene"):
            if mod in self.expert_dims.keys():
                experts = self.randomise_feats(experts, mod)
                aggregated_experts[mod] = experts[mod]

        for mod in ("audio", "speech", "ocr"):
            if mod in self.expert_dims.keys():
                experts[mod] = drop_nans(x=experts[mod], ind=ind[mod],
                                         validate_missing=True)
                experts = self.randomise_feats(experts, mod)
                aggregated_experts[mod] = self.pooling[mod](experts[mod])

        if "rgb" in self.expert_dims.keys():
            experts = self.randomise_feats(experts, "rgb")
            # If only average pooling has been performed, we will have an input of the
            # form N x 1 x D, so we need to flatten out the middle dimension to
            # maintain consistency
            aggregated_experts["rgb"] = experts["rgb"].view(experts["rgb"].shape[0], -1)

        # When pooling multiple captions for a single video, we treat them as separate
        # members of the minibatch, so the total pooling op does the following:
        # pooling: B x captions_per_video x max_sentence_length x text_feat_dim
        # -> B x captions_per_video (cluster_dim * text_feat_dim)
        B, captions_per_video, max_words, text_feat_dim = text.size()
        text = text.view(B * captions_per_video, max_words, text_feat_dim)
        text = self.text_pooling(text)
        text = text.view(B, captions_per_video, -1)
        return self.ce(text, aggregated_experts, ind, raw_captions)


class CEModule(nn.Module):
    def __init__(self, expert_dims, text_dim,
                 use_ce, verbose, l2renorm, disable_nan_checks,
                 random_feats, test_caption_mode, same_dim=512,
                 freeze_weights=False, keep_missing_modalities=False):
        super().__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.disable_nan_checks = disable_nan_checks
        self.test_caption_mode = test_caption_mode
        self.freeze_weights = freeze_weights
        self.random_feats = random_feats
        self.use_ce = use_ce
        self.verbose = verbose
        self.keep_missing_modalities = keep_missing_modalities
        self.l2renorm = l2renorm
        self.moe_fc = nn.Linear(text_dim, len(expert_dims))
        num_mods = len(expert_dims)
        self.moe_weights = th.ones(1, num_mods) / num_mods

        in_dims = [expert_dims[mod][0] for mod in modalities]
        agg_dims = [expert_dims[mod][1] for mod in modalities]

        # The batch size of the face input can vary (due to missing inputs), so we
        # probably shouldn't use BN on this branch. It's probably fine to leave it
        # n for the corresponding text inputs, (but we should switch to GN)
        use_bns = [True for modality in self.modalities]

        # NOTE: When use_ce is not used, the text features are projected to
        # subspaces of different dimensions.  When use_ce is used, they must all
        # be projected to `same_dim` (to allow fusion)
        if not self.use_ce:
            print("NOT use_ce")
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, dim, use_bn) for
                               in_dim, dim, use_bn in zip(in_dims, agg_dims, use_bns)]
            gated_text_embds = [GatedEmbeddingUnit(text_dim, dim, use_bn=True) for
                                dim in agg_dims]
        else:
            dim_reducers = [ReduceDim(in_dim, same_dim) for in_dim in in_dims]
            self.video_dim_reduce = nn.ModuleList(dim_reducers)

            self.g_reason_1 = nn.Linear(same_dim * 2, same_dim)
            self.g_reason_2 = nn.Linear(same_dim, same_dim)
            self.g_reason_3 = nn.Linear(same_dim, same_dim)

            self.f_reason_1 = nn.Linear(same_dim, same_dim)
            self.f_reason_2 = nn.Linear(same_dim, same_dim)
            self.f_reason_3 = nn.Linear(same_dim, same_dim)

            self.batch_norm_g1 = nn.BatchNorm1d(same_dim)
            self.batch_norm_g2 = nn.BatchNorm1d(same_dim)

            self.batch_norm_f1 = nn.BatchNorm1d(same_dim)
            self.batch_norm_f2 = nn.BatchNorm1d(same_dim)
            gated_vid_embds = [GatedEmbeddingUnitReasoning(same_dim) for _ in in_dims]
            gated_text_embds = [GatedEmbeddingUnit(text_dim, same_dim, use_bn=True) for
                                _ in modalities]

        self.video_GU = nn.ModuleList(gated_vid_embds)
        self.text_GU = nn.ModuleList(gated_text_embds)

    def compute_moe_weights(self, text, ind, freeze_weights):
        # compute weights for all captions (including when assigned K captions to
        # the same video)
        B, K, D = text.shape
        M = len(self.modalities)
        assert 1 <= M <= 7, "expected between 1 and 7 modalities"

        # Treat each caption independently in the softmax (which runs over modalities)
        text = text.view(B * K, D)
        if freeze_weights:
            moe_weights = self.moe_weights.repeat(B, K, 1)
            if text.is_cuda:
                moe_weights = moe_weights.cuda()
        else:
            moe_weights = self.moe_fc(text)  # BK x D -> BK x M
            moe_weights = F.softmax(moe_weights, dim=1)
            moe_weights = moe_weights.view(B, K, M)
        available = th.zeros_like(moe_weights)

        # mark locations of all missing
        if not self.keep_missing_modalities:
            for ii, modality in enumerate(self.modalities):
                available[:, :, ii] = ind[modality].view(-1, 1).repeat(1, K)
        else:
            available = th.ones_like(moe_weights)

        msg = "expected `available` modality mask to only contain 0s or 1s"
        assert set(th.unique(available).cpu().numpy()).issubset(set([0, 1])), msg

        if self.verbose:
            print("--------------------------------")
            for idx, key in enumerate(self.modalities):
                msg = "{}: mean: {:.3f}, std: {:.3f}, min: {:.3f}, max: {:.3f} [{}]"
                msg = msg.format(
                    key,
                    moe_weights[:, :, idx].mean().item(),
                    moe_weights[:, :, idx].std().item(),
                    moe_weights[:, :, idx].min().item(),
                    moe_weights[:, :, idx].max().item(),
                    available[:, :, idx].sum().item(),
                )
                print(msg)

        moe_weights = available * moe_weights
        norm_weights = th.sum(moe_weights, dim=2)
        norm_weights = norm_weights.view(B, K, 1)
        moe_weights = th.div(moe_weights, norm_weights)  # B x K x M
        self.last_smax_weights = moe_weights
        return moe_weights

    def mask_missing_embeddings(self, video, ind, exclude=[]):
        if self.disable_nan_checks:
            validate_missing = False
        else:
            validate_missing = True
        for modality in self.modalities:
            # swap out NaNs for zeros in each set of features and apply safety check
            if modality in exclude:
                continue
            video[modality] = drop_nans(
                x=video[modality],
                ind=ind[modality],
                validate_missing=validate_missing,
            )
        return video

    def forward(self, text, experts, ind, raw_captions):
        """Compute joint embeddings and, if requested, a confusion matrix between
        video and text representations in the minibatch.

        Notation: B = batch size, M = number of modalities
        """
        # Pass text embeddings through gated units
        text_embd = {}

        # Unroll repeated captions into present minibatch
        B, captions_per_video, feat_dim = text.size()
        text = text.view(B * captions_per_video, feat_dim)

        for modality, layer in zip(self.modalities, self.text_GU):
            # NOTE: Due to the batch norm, the gated units are sensitive to passing
            # in a lot of zeroes, so we do the masking step after the forwards pass
            text_ = layer(text)
            text_ = text_.view(B, captions_per_video, -1)
            text_ = drop_nans(text_, ind[modality], validate_missing=False)
            if "text" in self.random_feats:
                text_ = th.rand_like(text_)
            text_embd[modality] = text_
        text = text.view(B, captions_per_video, -1)

        # speech/ocr/audio nans are handled earlier (during pooling)
        exclude = ["speech", "ocr", "audio"]

        # avoid zeroing random features, since this will leak information
        exclude = exclude + list(self.random_feats)
        experts = self.mask_missing_embeddings(experts, ind, exclude=exclude)

        # MOE weights computation + normalization - note that we use the first caption
        # sample to predict the weights
        moe_weights = self.compute_moe_weights(text, ind=ind,
                                               freeze_weights=self.freeze_weights)

        # Embed all features to a common dimension
        if not self.use_ce:
            for modality, layer in zip(self.modalities, self.video_GU):
                experts[modality] = layer(experts[modality])
        else:
            for modality, layer in zip(self.modalities, self.video_dim_reduce):
                experts[modality] = layer(experts[modality])

            all_combinations = list(itertools.permutations(experts, 2))
            assert len(self.modalities) > 1, "use_ce requires multiple modalities"

            for ii, l in enumerate(self.video_GU):

                mask_num = 0
                curr_mask = 0
                temp_dict = {}
                avai_dict = {}
                curr_modality = self.modalities[ii]

                for modality_pair in all_combinations:
                    mod0, mod1 = modality_pair
                    if mod0 == curr_modality:
                        new_key = "_".join(modality_pair)
                        fused = th.cat((experts[mod0], experts[mod1]), 1)  # -> B x 2D
                        temp = self.g_reason_1(fused)  # B x 2D -> B x D
                        # temp=self.batch_norm_g(temp)
                        temp = self.g_reason_2(F.relu(temp))  # B x D -> B x D
                        # temp=self.g_reason_3(F.relu(temp))
                        temp_dict[new_key] = temp
                        avail = (ind[mod0].float() * ind[mod1].float()).to(text.device)
                        avai_dict[new_key] = avail

                # Combine the paired features into a mask through elementwise summation
                for mm in temp_dict:
                    curr_mask += temp_dict[mm] * avai_dict[mm].unsqueeze(1)
                    mask_num += avai_dict[mm]

                curr_mask = th.div(curr_mask, (mask_num + 0.00000000001).unsqueeze(1))
                curr_mask = self.f_reason_1(curr_mask)
                # curr_mask = self.batch_norm_f(curr_mask)
                curr_mask = self.f_reason_2(F.relu(curr_mask))
                # curr_mask = self.f_reason_3 (F.relu(curr_mask))
                # curr_mask= F.relu （curr_mask）
                experts[curr_modality] = l(experts[curr_modality], curr_mask)

        if self.training:
            merge_caption_similiarities = "avg"
        else:
            merge_caption_similiarities = self.test_caption_mode

        # hard code for safety
        # merge_caption_similiarities = "indep"

        cross_view_conf_matrix = sharded_cross_view_inner_product(
            vid_embds=experts,
            text_embds=text_embd,
            l2renorm=self.l2renorm,
            text_weights=moe_weights,
            subspaces=self.modalities,
            raw_captions=raw_captions,
            merge_caption_similiarities=merge_caption_similiarities,
        )
        return {
            "modalities": self.modalities,
            "cross_view_conf_matrix": cross_view_conf_matrix,
        }


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class ReduceDim(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedEmbeddingUnitReasoning(nn.Module):
    def __init__(self, output_dimension):
        super(GatedEmbeddingUnitReasoning, self).__init__()
        self.cg = ContextGatingReasoning(output_dimension)

    def forward(self, x, mask):
        x = self.cg(x, mask)
        x = F.normalize(x)
        return x


class ContextGatingReasoning(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGatingReasoning, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        self.batch_norm2 = nn.BatchNorm1d(dimension)

    def forward(self, x, x1):

        x2 = self.fc(x)

        # t = x1 + x2
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
            x2 = self.batch_norm2(x2)
            # t = self.batch_norm (t)

        # t = (F.sigmoid(x1) + F.sigmoid (x2))/2

        t = x1 + x2

        # t = (t > 0.2).float() * 1
        # t = th.trunc(2*F.sigmoid (t)-0.5)
        # print (t)
        # return x*F.sigmoid(t)

        # return t  (curr no sigmoid hoho!)
        x = th.cat((x, t), 1)
        return F.glu(x, 1)


def sharded_cross_view_inner_product(vid_embds, text_embds, text_weights,
                                     subspaces, l2renorm,
                                     merge_caption_similiarities="avg", tol=1E-5,
                                     raw_captions=None):
    """Compute a similarity matrix from sharded vectors.

    Args:
        embds1 (dict[str:th.Tensor]): the set of sub-embeddings that, when
            concatenated, form the whole. The ith shard has shape `B x K x F_i`
            (i.e. they can differ in the last dimension).
        embds2 (dict[str:th.Tensor]): same format.
        weights2 (th.Tensor): weights for the shards in `embds2`.
        l2norm (bool::True): whether to l2 renormalize the full embeddings.

    Returns:
        (th.tensor): similarity matrix of size `BK x BK`.

    NOTE: If multiple captions are provided, we can aggregate their similarities to
    provide a single video-text similarity score.
    """
    B = vid_embds[subspaces[0]].size(0)
    device = vid_embds[subspaces[0]].device
    num_caps = text_embds[subspaces[0]].size(1)

    # unroll separate captions onto first dimension and treat them separately
    sims = th.zeros(B * num_caps, B, device=device)
    text_weights = text_weights.view(B * num_caps, -1)

    # if only_use_first_caption:
    #     caption_idx = 0
    # else:
    #     raise NotImplementedError("only using the first caption is supported")

    if l2renorm:
        l2_mass_vid, l2_mass_text = 0, 0
        for idx, modality in enumerate(subspaces):
            vid_embd_ = vid_embds[modality]
            assert len(vid_embd_.size()) == 2, "expected B x feat_dim format"
            l2_mass_vid += vid_embd_.reshape(B, -1).pow(2).sum(1)
            text_embd_ = text_embds[modality]
            assert len(text_embd_.size()) == 3, "expected B x caps x feat_dim format"
            text_embd_ = text_embd_.reshape(B * num_caps, -1)
            text_embd_ = text_weights[:, idx:idx + 1] * text_embd_
            l2_mass_text += text_embd_.pow(2).sum(1)
        l2_mass_vid = th.sqrt(l2_mass_vid.clamp(min=1E-6)).unsqueeze(1)
        l2_mass_text = th.sqrt(l2_mass_text.clamp(min=1E-6)).unsqueeze(1)
    else:
        l2_mass_text, l2_mass_vid = 1, 1

    for idx, modality in enumerate(subspaces):
        vid_embd_ = vid_embds[modality].reshape(B, -1) / l2_mass_vid
        text_embd_ = text_embds[modality].view(B * num_caps, -1)
        text_embd_ = text_weights[:, idx: idx + 1] * text_embd_
        msg = "expected weights to be applied to text embeddings"
        assert text_embd_.shape[0] == text_weights.shape[0], msg
        text_embd_ = text_embd_ / l2_mass_text
        sims += th.matmul(text_embd_, vid_embd_.t())  # (B x num_caps) x (B)

    if l2renorm:
        # if not (sims.max() < 1 + tol):
        #     import ipdb; ipdb.set_trace()
        assert sims.max() < 1 + tol, "expected cosine similarities to be < 1"
        assert sims.min() > -1 - tol, "expected cosine similarities to be > -1"

    if th.isnan(sims).sum().item():
        raise ValueError("Found nans in similarity matrix!")

    if num_caps > 1:
        # aggregate similarities from different captions
        if merge_caption_similiarities == "avg":
            sims = sims.view(B, num_caps, B)
            # vis = False
            # if vis:
            #     import sys
            #     from pathlib import Path
            #     import matplotlib
            #     matplotlib.use("Agg")
            #     import matplotlib.pyplot as plt
            #     sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
            #     from zsvision.zs_iterm import zs_dispFig # NOQA
            #     sim_stds = th.std(sims, dim=1)
            #     sim_stds_ = sim_stds - sim_stds.min()
            #     sim_stds_ = sim_stds_ / sim_stds_.max()
            #     plt.matshow(sim_stds_)
            #     zs_dispFig()
            #     import ipdb; ipdb.set_trace()

            sims = th.mean(sims, dim=1)
            sims = sims.view(B, B)
        elif merge_caption_similiarities == "indep":
            pass
        else:
            msg = "unrecognised merge mode: {}"
            raise ValueError(msg.format(merge_caption_similiarities))
    return sims


def sharded_single_view_inner_product(embds, subspaces, text_weights=None,
                                      l2renorm=True):
    """Compute a similarity matrix from sharded vectors.

    Args:
        embds (dict[str:th.Tensor]): the set of sub-embeddings that, when concatenated,
            form the whole. The ith shard has shape `B x K x F_i` (i.e. they can
            differ in the last dimension), or shape `B x F_i`
        l2norm (bool::True): whether to l2 normalize the full embedding.

    Returns:
        (th.tensor): similarity matrix of size `BK x BK`.
    """
    subspaces = list(embds.keys())
    device = embds[subspaces[0]].device
    shape = embds[subspaces[0]].shape
    if len(shape) == 3:
        B, K, _ = shape
        num_embds = B * K
        assert text_weights is not None, "Expected 3-dim tensors for text (+ weights)"
        assert text_weights.shape[0] == B
        assert text_weights.shape[1] == K
    elif len(shape) == 2:
        B, _ = shape
        num_embds = B
        assert text_weights is None, "Expected 2-dim tensors for non-text (no weights)"
    else:
        raise ValueError("input tensor with {} dims unrecognised".format(len(shape)))
    sims = th.zeros(num_embds, num_embds, device=device)
    if l2renorm:
        l2_mass = 0
        for idx, modality in enumerate(subspaces):
            embd_ = embds[modality]
            if text_weights is not None:
                # text_weights (i.e. moe_weights) are shared among subspace for video
                embd_ = text_weights[:, :, idx:idx + 1] * embd_
            embd_ = embds[modality].reshape(num_embds, -1)
            l2_mass += embd_.pow(2).sum(1)
        l2_mass = th.sqrt(l2_mass.clamp(min=1E-6)).unsqueeze(1)
    else:
        l2_mass = 1

    for idx, modality in enumerate(subspaces):
        embd_ = embds[modality]
        if text_weights is not None:
            embd_ = text_weights[:, :, idx:idx + 1] * embd_
        embd_ = embd_.reshape(num_embds, -1) / l2_mass
        sims += th.matmul(embd_, embd_.t())
    if th.isnan(sims).sum().item():
        # import ipdb; ipdb.set_trace()
        raise ValueError("Found nans in similarity matrix!")
    return sims


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import argparse
    parser = argparse.ArgumentParser(description="Video Retrieval")
    parser.add_argument("--single_view", action="store_true")
    parser.add_argument("--cross_view", action="store_true")
    parser.add_argument("--modality_norm", action="store_true")
    parser.add_argument("--l2renorm", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
    from zsvision.zs_iterm import zs_dispFig  # NOQA
    n = 5
    captions_per_vid = 3

    # should be basically orthogonal outside the bands
    sc = 1 / 3
    featdim = 10

    th.manual_seed(0)
    src = th.randn(n, 1, featdim)
    noise = sc * th.randn(n, captions_per_vid, featdim)
    src_a = src.repeat(1, captions_per_vid, 1) + noise

    src = th.randn(n, 1, featdim)
    noise = sc * th.randn(n, captions_per_vid, featdim)
    src_b = src.repeat(1, captions_per_vid, 1) + noise

    sample_modalities = ["a", "b"]
    if args.modality_norm:
        src_a = F.normalize(src_a, dim=2)
        src_b = F.normalize(src_b, dim=2)
    text_embds = {"a": src_a, "b": src_b}
    vid_embds = {"a": src_a[:, 0, :], "b": src_b[:, 0, :]}
    weights = 0.5 * th.ones(n, captions_per_vid, len(sample_modalities))

    if args.single_view:
        conf_mat = sharded_single_view_inner_product(
            embds=vid_embds,
            text_weights=None,
            subspaces=sample_modalities,
        )
        plt.matshow(conf_mat)
        zs_dispFig()

        conf_mat = sharded_single_view_inner_product(
            embds=text_embds,
            text_weights=weights,
            subspaces=sample_modalities,
        )
        plt.matshow(conf_mat)
        zs_dispFig()

    if args.cross_view:
        conf_mat = sharded_cross_view_inner_product(
            vid_embds=vid_embds,
            l2renorm=args.l2renorm,
            text_embds=text_embds,
            text_weights=weights,
            subspaces=sample_modalities,
        )
        plt.matshow(conf_mat)
        zs_dispFig()
