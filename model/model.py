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

import ipdb
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from base import BaseModel
from utils import expert_tensor_storage
from model.net_vlad import NetVLAD


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    SRC: https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
    '''
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * th.tanh(F.softplus(input))


def kronecker_prod(t1, t2):
    # kronecker is performed along the last dim
    kron = th.bmm(t1.view(-1, t1.size(-1), 1), t2.contiguous().view(-1, 1, t2.size(-1)))
    return kron.view(t1.shape[0], t1.shape[1], -1)


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
        x = x_
    if th.isnan(x).sum() > 0:
        # slow, but might help us solve the mystery
        ipdb.set_trace()
    return x


class CENet(BaseModel):
    def __init__(
            self,
            task,
            use_ce,
            text_dim,
            l2renorm,
            expert_dims,
            vlad_clusters,
            ghost_clusters,
            disable_nan_checks,
            keep_missing_modalities,
            test_caption_mode,
            randomise_feats,
            feat_aggregation,
            ce_shared_dim,
            trn_config,
            trn_cat,
            include_self,
            use_mish,
            use_bn_reason,
            num_h_layers,
            num_g_layers,
            kron_dets=False,
            freeze_weights=False,
            geometric_mlp=False,
            rand_proj=False,
            mimic_ce_dims=False,
            coord_dets=False,
            concat_experts=False,
            spatial_feats=False,
            concat_mix_experts=False,
            verbose=False,
            num_classes=None):
        super().__init__()

        self.l2renorm = l2renorm
        self.task = task
        self.geometric_mlp = geometric_mlp
        self.feat_aggregation = feat_aggregation
        self.expert_dims = expert_dims
        self.num_h_layers = num_h_layers
        self.num_g_layers = num_g_layers
        self.use_mish = use_mish
        self.use_bn_resaon = use_bn_reason
        self.include_self = include_self
        self.kron_dets = kron_dets
        self.rand_proj = rand_proj
        self.coord_dets = coord_dets
        self.disable_nan_checks = disable_nan_checks
        self.trn_config = trn_config
        self.trn_cat = trn_cat
        if randomise_feats:
            self.random_feats = set([x for x in randomise_feats.split(",")])
        else:
            self.random_feats = set()

        # sanity checks on the features that may be vladded
        pre_vlad_feat_sizes = {"ocr": 300, "audio": 128, "speech": 300}
        pre_vlad_feat_sizes = {key: val for key, val in pre_vlad_feat_sizes.items()
                               if feat_aggregation[key]["temporal"] == "vlad"}

        # we basically disable safety checks for detection-sem
        if spatial_feats:
            spatial_feat_dim = 16
        else:
            spatial_feat_dim = 5
        if self.geometric_mlp:
            self.geometric_mlp_model = SpatialMLP(spatial_feat_dim)
        if kron_dets:
            sem_det_dim = 300 * spatial_feat_dim
        elif coord_dets:
            sem_det_dim = spatial_feat_dim
        elif rand_proj:
            sem_det_dim = 300 + 300
            self.proj = nn.Linear(spatial_feat_dim, 300)
            # random proj
            # th.nn.init.normal_(self.proj.weight, mean=0, std=1)
            # for param in self.proj.parameters():
            #     param.requires_grad = False
            # self.proj.bias.mul_(0)
        else:
            sem_det_dim = 300 + spatial_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        pre_vlad_feat_sizes["detection-sem"] = sem_det_dim
        if "detection-sem" in expert_dims:
            new_in_dim = sem_det_dim * vlad_clusters["detection-sem"]
            expert_dims["detection-sem"] = (new_in_dim, expert_dims["detection-sem"][1])

        vlad_feat_sizes = {key: val for key, val in vlad_clusters.items()}

        self.pooling = nn.ModuleDict()
        for mod, expected in pre_vlad_feat_sizes.items():
            if mod in expert_dims.keys():
                feature_size = expert_dims[mod][0] // vlad_clusters[mod]
                msg = f"expected {expected} for {mod} features atm"
                assert feature_size == expected, msg
                self.pooling[mod] = NetVLAD(
                    feature_size=feature_size,
                    cluster_size=vlad_clusters[mod],
                )
        if "retrieval" in self.task:
            if vlad_clusters["text"] == 0:
                self.text_pooling = nn.Sequential()
            else:
                self.text_pooling = NetVLAD(
                    feature_size=text_dim,
                    cluster_size=vlad_clusters["text"],
                    ghost_clusters=ghost_clusters["text"],
                )
                text_dim = self.text_pooling.out_dim
        else:
            self.num_classes = num_classes
            text_dim = None

        self.tensor_storage = expert_tensor_storage(
            experts=self.expert_dims.keys(),
            feat_aggregation=self.feat_aggregation,
        )
        # self.fixed_sz_experts = ["rgb", "scene", "face", "flow"]
        # self.variable_sz_experts = ["audio", "speech", "ocr"]
        # # handle features which can be either
        # for expert in {"detection", "detection-sem", "openpose"}:
        #     if expert in self.pooling:
        #         self.variable_sz_experts.append(expert)
        #     else:
        #         self.fixed_sz_experts.append(expert)

        self.ce = CEModule(
            use_ce=use_ce,
            task=self.task,
            verbose=verbose,
            l2renorm=l2renorm,
            trn_cat=self.trn_cat,
            trn_config=self.trn_config,
            random_feats=self.random_feats,
            freeze_weights=freeze_weights,
            text_dim=text_dim,
            test_caption_mode=test_caption_mode,
            concat_experts=concat_experts,
            concat_mix_experts=concat_mix_experts,
            expert_dims=expert_dims,
            vlad_feat_sizes=vlad_feat_sizes,
            disable_nan_checks=disable_nan_checks,
            keep_missing_modalities=keep_missing_modalities,
            mimic_ce_dims=mimic_ce_dims,
            include_self=include_self,
            use_mish=use_mish,
            use_bn_reason=use_bn_reason,
            num_h_layers=num_h_layers,
            num_g_layers=num_g_layers,
            num_classes=num_classes,
            same_dim=ce_shared_dim,
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

    def forward(self, experts, ind, text=None, raw_captions=None, text_token_mask=None):
        aggregated_experts = OrderedDict()

        if "detection-sem" in self.expert_dims:
            det_sem = experts["detection-sem"]
            box_feats = det_sem[:, :, :self.spatial_feat_dim]
            sem_feats = det_sem[:, :, self.spatial_feat_dim:]
            if self.geometric_mlp:
                x = box_feats.view(-1, box_feats.shape[-1])
                x = self.geometric_mlp_model(x)
                box_feats = x.view(box_feats.shape)

            if self.kron_dets:
                feats = kronecker_prod(box_feats, sem_feats)
            elif self.coord_dets:
                feats = box_feats.contiguous()
            elif self.rand_proj:
                feats = box_feats.contiguous()
                projected = self.proj(feats)
                feats = th.cat((projected, sem_feats.contiguous()), dim=2)
            else:
                feats = th.cat((box_feats, sem_feats.contiguous()), dim=2)
            experts["detection-sem"] = feats

        # Handle all nan-checks
        for mod in self.expert_dims:
            experts = self.randomise_feats(experts, mod)
            experts[mod] = drop_nans(x=experts[mod], ind=ind[mod], validate_missing=True)
            if mod in self.tensor_storage["fixed"]:
                aggregated_experts[mod] = experts[mod]
            elif mod in self.tensor_storage["variable"]:
                aggregated_experts[mod] = self.pooling[mod](experts[mod])

        # for mod in variable_sz_experts:
        #     if mod in self.expert_dims.keys():
        #         experts[mod] = drop_nans(x=experts[mod], ind=ind[mod],
        #                                  validate_missing=True)
        #         experts = self.randomise_feats(experts, mod)

        # if "rgb" in self.expert_dims.keys():
        #     experts = self.randomise_feats(experts, "rgb")
        #     # If only average pooling has been performed, we will have an input of the
        #     # form N x 1 x D, so we need to flatten out the middle dimension to
        #     # maintain consistency
        #     aggregated_experts["rgb"] = experts["rgb"].view(experts["rgb"].shape[0], -1)

        if "retrieval" in self.task:
            # When pooling multiple captions for a single video, we treat them as separate
            # members of the minibatch, so the total pooling op does the following:
            # pooling: B x captions_per_video x max_sentence_length x text_feat_dim
            # -> B x captions_per_video (cluster_dim * text_feat_dim)
            B, captions_per_video, max_words, text_feat_dim = text.size()
            text = text.view(B * captions_per_video, max_words, text_feat_dim)

            # avoid OOM errors on old GPUs during inference
            # if not self.text_pooling.training:
            #     safe = 5000
            #     if text.shape[0] > safe:
            #         splits = [safe] * (text.shape[0] // safe)
            #         splits += [text.shape[0] - sum(splits)]
            #         texts = th.split(text, split_size_or_sections=splits)
            #         texts = [self.text_pooling(text) for text in texts]
            #         text = th.cat(texts, dim=0)
            #         import ipdb; ipdb.set_trace()
            #     else:
            # safe = 5000
            # if text.shape[0] > safe:
            #     dev = text.device
            #     print("pooling text features on cpu")
            #     text, self.text_pooling = text.to("cpu"), self.text_pooling.to("cpu")
            #     text = self.text_pooling(text)
            #     text, self.text_pooling = text.to(dev), self.text_pooling.to(dev)
            #     import ipdb; ipdb.set_trace()
            # else:
            if isinstance(self.text_pooling, NetVLAD):
                kwargs = {"mask": text_token_mask}
            else:
                kwargs = {}
            text = self.text_pooling(text, **kwargs)
            text = text.view(B, captions_per_video, -1)
        else:
            text = None
        return self.ce(text, aggregated_experts, ind, raw_captions)


class TemporalAttention(th.nn.Module):
    def __init__(self, img_feature_dim, num_attention):
        super().__init__()
        self.weight = Variable(
            th.randn(img_feature_dim, num_attention),
            requires_grad=True).cuda()  # d*seg
        self.img_feature_dim = img_feature_dim
        self.num_attention = num_attention

    def forward(self, input):
        B, T, D = input.shape
        record = []
        input_avg = th.mean(input.clone(), dim=1)
        input_max = th.max(input.clone(), dim=1)
        record.append(input_avg)
        record.append(input_max[0])
        output = th.matmul(input, self.weight)
        attentions = F.softmax(output, dim=1)
        for idx in range(attentions.shape[-1]):
            temp = attentions[:, :, idx]
            temp_output = th.sum(temp.unsqueeze(2) * input, dim=1)
            norm = temp_output.norm(p=2, dim=-1, keepdim=True)
            temp_output = temp_output.div(norm)
            record.append(temp_output)
        act_all = th.cat((record), 1)
        return act_all


class RelationModuleMultiScale(th.nn.Module):
    # Temporal Relation module in multiply scale, suming over
    # [2-frame relation, 3-frame relation, ..., n-frame relation]
    def __init__(self, img_feature_dim, num_frames, num_class):
        super().__init__()
        self.subsample_num = 3  # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        # generate the multiple frame relations
        self.scales = [i for i in range(num_frames, 1, -1)]

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            # how many samples of relation to select in each forward pass
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
            )
            self.fc_fusion_scales += [fc_fusion]

        msg = 'Multi-Scale Temporal Relation Network Module in use'
        print(msg, ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(
                len(self.relations_scales[scaleID]),
                self.subsample_scales[scaleID],
                replace=False,
            )
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class RelationModuleMultiScale_Cat(th.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale_Cat, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        record = []
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        norm = act_all.norm(p=2, dim=-1, keepdim=True)
        act_all = act_all.div(norm)
        record.append(act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            act_all = 0
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
            norm = act_all.norm(p=2, dim=-1, keepdim=True)
            act_all = act_all.div(norm)
            record.append(act_all)

        act_all = th.cat((record), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], 
                    num_frames_relation))


class CEModule(nn.Module):
    def __init__(self, expert_dims, text_dim, use_ce, verbose, l2renorm, num_classes,
                 trn_config, trn_cat, use_mish, include_self, num_h_layers, num_g_layers,
                 disable_nan_checks, random_feats, test_caption_mode, mimic_ce_dims,
                 concat_experts, concat_mix_experts, freeze_weights, task,
                 keep_missing_modalities, vlad_feat_sizes, same_dim, use_bn_reason):
        super().__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.disable_nan_checks = disable_nan_checks
        self.mimic_ce_dims = mimic_ce_dims
        self.concat_experts = concat_experts
        self.same_dim = same_dim
        self.use_mish = use_mish
        self.use_bn_reason = use_bn_reason
        self.num_h_layers = num_h_layers
        self.num_g_layers = num_g_layers
        self.include_self = include_self
        self.num_classes = num_classes
        self.task = task
        self.vlad_feat_sizes = vlad_feat_sizes
        self.concat_mix_experts = concat_mix_experts
        self.test_caption_mode = test_caption_mode
        self.reduce_dim = 64
        self.moe_cg = ContextGating
        self.freeze_weights = freeze_weights
        self.random_feats = random_feats
        self.use_ce = use_ce
        self.verbose = verbose
        self.keep_missing_modalities = keep_missing_modalities
        self.l2renorm = l2renorm
        self.trn_config = trn_config
        self.trn_cat = trn_cat
        print("trn_config is {}".format(self.trn_config))

        if self.use_mish:
            self.non_lin = Mish()
        else:
            self.non_lin = nn.ReLU()

        if "retrieval" in self.task:
            num_mods = len(expert_dims)
            self.moe_fc = nn.Linear(text_dim, len(expert_dims))
            # self.moe_fc_bottleneck1 = nn.Linear(text_dim, text_dim // 4)
            # self.moe_cg = nn.Linear(text_dim // 4, text_dim // 4)
            # self.moe_fc_proj = nn.Linear(text_dim // 4, len(expert_dims))
            self.moe_weights = th.ones(1, num_mods) / num_mods

        # The batch size of the face input can vary (due to missing inputs), so we
        # probably shouldn't use BN on this branch. It's probably fine to leave it
        # n for the corresponding text inputs, (but we should switch to GN)
        use_bns = [True for modality in self.modalities]

        # NOTE: When use_ce is not used, the text features are projected to
        # subspaces of different dimensions.  When use_ce is used, they must all
        # be projected to `same_dim` (to allow fusion). The only excpetion is for an
        # ablation in which we mimic the `same_dim` reduction to measure whether this
        # projection influences overall performance.
        self.trn_list = nn.ModuleList()

        self.repeat_temporal = {}
        for mod in modalities:
            self.repeat_temporal[mod] = 1

        if self.trn_cat == 2:
            print("Performing concat between random temporal attention")
            for mod in self.trn_config.keys():
                img_feature_dim = expert_dims[mod][0]  # 365
                num_frames = self.trn_config[
                    mod]  # This is exatcly how many different attention
                num_frames = 1  # mimic simple avg and max based on segments
                # num_class = expert_dims[mod][0]
                self.trn_list += [TemporalAttention(img_feature_dim, num_frames)]
                self.repeat_temporal[mod] = num_frames + 2
        elif self.trn_cat == 1:
            print("Performing concat between segments")
            for mod in self.trn_config.keys():
                img_feature_dim = expert_dims[mod][0]  # 365
                num_frames = self.trn_config[mod]  # hard code
                num_class = expert_dims[mod][0]
                self.trn_list += [
                    RelationModuleMultiScale_Cat(img_feature_dim, num_frames, num_class)
                ]
                self.repeat_temporal[mod] = len(
                    [i for i in range(num_frames, 1, -1)])
        elif self.trn_cat == 0:
            print("Performing Conventional TRN (sum) segments")
            for mod in self.trn_config.keys():
                img_feature_dim = expert_dims[mod][0]  # 365
                num_frames = self.trn_config[mod]  # hard code
                num_class = expert_dims[mod][0]
                self.trn_list += [
                    RelationModuleMultiScale(img_feature_dim, num_frames,
                                             num_class)
                ]
        else:
            raise NotImplementedError()

        in_dims = [expert_dims[mod][0] * self.repeat_temporal[mod] for mod in modalities]
        agg_dims = [expert_dims[mod][1] * self.repeat_temporal[mod] for mod in modalities]

        if self.use_ce or self.mimic_ce_dims:
            dim_reducers = [ReduceDim(in_dim, same_dim) for in_dim in in_dims]
            self.video_dim_reduce = nn.ModuleList(dim_reducers)
        if self.use_ce:
            # The g_reason module has a first layer that is specific to the design choice
            # (e.g. triplet vs pairwise), then a shared component which is common to all
            # designs.
            if self.use_ce in {"pairwise", "pairwise-star", "triplet"}:
                num_inputs = 3 if self.use_ce == "triplet" else 2
                self.g_reason_1 = nn.Linear(same_dim * num_inputs, same_dim)
            elif self.use_ce == "pairwise-star-specific":
                num_inputs = 2
                g_reason_unshared_weights = [G_reason(same_dim, num_inputs, self.non_lin)
                                             for mod in modalities]
                self.g_reason_unshared_weights = nn.ModuleList(g_reason_unshared_weights)
            elif self.use_ce in {"pairwise-star-tensor"}:
                reduce_dim = self.reduce_dim
                self.dim_reduce = nn.Linear(same_dim, reduce_dim)
                self.g_reason_1 = nn.Linear(self.reduce_dim * reduce_dim, same_dim)
            else:
                raise ValueError(f"unrecognised CE config: {self.use_ce}")

            g_reason_shared = []
            for _ in range(self.num_g_layers - 1):
                if self.use_bn_reason:
                    g_reason_shared.append(nn.BatchNorm1d(same_dim))
                g_reason_shared.append(self.non_lin)
                g_reason_shared.append(nn.Linear(same_dim, same_dim))
            self.g_reason_shared = nn.Sequential(*g_reason_shared)

            h_reason = []
            for _ in range(self.num_h_layers):
                if self.use_bn_reason:
                    h_reason.append(nn.BatchNorm1d(same_dim))
                h_reason.append(self.non_lin)
                h_reason.append(nn.Linear(same_dim, same_dim))
            self.h_reason = nn.Sequential(*h_reason)

            gated_vid_embds = [GatedEmbeddingUnitReasoning(same_dim) for _ in in_dims]
            text_out_dims = [same_dim for _ in agg_dims]
        elif self.mimic_ce_dims:  # ablation study
            gated_vid_embds = [MimicCEGatedEmbeddingUnit(same_dim, same_dim, use_bn=True)
                               for _ in modalities]
            text_out_dims = [same_dim for _ in agg_dims]
        elif self.concat_mix_experts:  # ablation study
            # use a single large GEU to mix the experts - the output will be the sum
            # of the aggregation sizes
            in_dim, out_dim = sum(in_dims), sum(agg_dims)
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, out_dim, use_bn=True)]
        elif self.concat_experts:  # ablation study
            # We do not use learnable parameters for the video combination, (we simply
            # use a high dimensional inner product).
            gated_vid_embds = []
        else:
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, dim, use_bn) for
                               in_dim, dim, use_bn in zip(in_dims, agg_dims, use_bns)]
            text_out_dims = agg_dims
        self.video_GU = nn.ModuleList(gated_vid_embds)

        if "retrieval" in self.task:
            if self.concat_experts:
                gated_text_embds = [nn.Sequential()]
            elif self.concat_mix_experts:
                # As with the video inputs, we similiarly use a single large GEU for the
                # text embedding
                gated_text_embds = [GatedEmbeddingUnit(text_dim, sum(agg_dims),
                                    use_bn=True)]
            else:
                gated_text_embds = [GatedEmbeddingUnit(text_dim, dim, use_bn=True) for
                                    dim in text_out_dims]
            self.text_GU = nn.ModuleList(gated_text_embds)
        else:
            print("V. simple classifier, should update....")
            total_dim = 0
            for mod in self.expert_dims.keys():
                total_dim += self.expert_dims[mod][1] * self.repeat_temporal[mod]
            print(f"Total dim is {total_dim}")
            self.classifier = nn.Linear(total_dim, self.num_classes)

    def compute_moe_weights(self, text, ind):
        # compute weights for all captions (including when assigned K captions to
        # the same video)
        B, K, D = text.shape
        M = len(self.modalities)
        msg = f"expected between 1 and 10 modalities, found {M} ({self.modalities})"
        assert 1 <= M <= 10, msg

        # Treat each caption independently in the softmax (which runs over modalities)
        text = text.view(B * K, D)
        if self.freeze_weights:
            moe_weights = self.moe_weights.repeat(B, K, 1)
            if text.is_cuda:
                moe_weights = moe_weights.cuda()
        else:
            # if False:
            #     print("USING BIGGER WEIGHT PREDS")
            #     moe_weights = self.moe_fc_bottleneck1(text)
            #     moe_weights = self.moe_cg(moe_weights)
            #     moe_weights = self.moe_fc_proj(moe_weights)
            #     moe_weights = moe_weights * 1
            # else:
            moe_weights = self.moe_fc(text)  # BK x D -> BK x M
            moe_weights = F.softmax(moe_weights, dim=1)
            moe_weights = moe_weights.view(B, K, M)

        if self.verbose:
            print("--------------------------------")
            for idx, key in enumerate(self.modalities):
                msg = "{}: mean: {:.3f}, std: {:.3f}, min: {:.3f}, max: {:.3f}"
                msg = msg.format(
                    key,
                    moe_weights[:, :, idx].mean().item(),
                    moe_weights[:, :, idx].std().item(),
                    moe_weights[:, :, idx].min().item(),
                    moe_weights[:, :, idx].max().item(),
                )
                print(msg)
        return moe_weights

    def forward(self, text, experts, ind, raw_captions):
        """Compute joint embeddings and, if requested, a confusion matrix between
        video and text representations in the minibatch.

        Notation: B = batch size, M = number of modalities
        """
        if "retrieval" in self.task:
            # Pass text embeddings through gated units
            text_embd = {}

            # Unroll repeated captions into present minibatch
            B, captions_per_video, feat_dim = text.size()
            text = text.view(B * captions_per_video, feat_dim)
            for modality, layer in zip(self.modalities, self.text_GU):
                # NOTE: Due to the batch norm, the gated units are sensitive to passing
                # in a lot of zeroes, so we do the masking step after the forwards pass
                text_ = layer(text)

                # We always assume that text is available for retrieval
                text_ = text_.view(B, captions_per_video, -1)

                if "text" in self.random_feats:
                    text_ = th.rand_like(text_)
                text_embd[modality] = text_
            text = text.view(B, captions_per_video, -1)

            # vladded nans are handled earlier (during pooling)
            # We also avoid zeroing random features, since this will leak information
            # exclude = list(self.vlad_feat_sizes.keys()) + list(self.random_feats)
            # experts = self.mask_missing_embeddings(experts, ind, exclude=exclude)

            # MOE weights computation + normalization - note that we use the first caption
            # sample to predict the weights
            moe_weights = self.compute_moe_weights(text, ind=ind)

        if self.l2renorm:
            for modality in self.modalities:
                norm = experts[modality].norm(p=2, dim=-1, keepdim=True)
                experts[modality] = experts[modality].div(norm)

        for modality, layer in zip(self.modalities, self.trn_list):
            experts[modality] = layer(experts[modality])

        if hasattr(self, "video_dim_reduce"):
            # Embed all features to a common dimension
            for modality, layer in zip(self.modalities, self.video_dim_reduce):
                experts[modality] = layer(experts[modality])

        if self.use_ce:
            dev = experts[self.modalities[0]].device
            if self.include_self:
                all_combinations = list(itertools.product(experts, repeat=2))
            else:
                all_combinations = list(itertools.permutations(experts, 2))
            assert len(self.modalities) > 1, "use_ce requires multiple modalities"

            if self.use_ce in {"pairwise-star", "pairwise-star-specific",
                               "pairwise-star-tensor"}:
                sum_all = 0
                sum_ind = 0
                for mod0 in experts.keys():
                    sum_all += (experts[mod0] * ind[mod0].float().to(dev).unsqueeze(1))
                    sum_ind += ind[mod0].float().to(dev).unsqueeze(1)
                avg_modality = sum_all / sum_ind

            for ii, l in enumerate(self.video_GU):

                mask_num = 0
                curr_mask = 0
                temp_dict = {}
                avai_dict = {}
                curr_modality = self.modalities[ii]

                if self.use_ce == "pairwise-star":
                    fused = th.cat((experts[curr_modality], avg_modality), 1)  # -> B x 2D
                    temp = self.g_reason_1(fused)  # B x 2D -> B x D
                    temp = self.g_reason_shared(temp)  # B x D -> B x D
                    curr_mask = temp * ind[curr_modality].float().to(dev).unsqueeze(1)

                elif self.use_ce == "pairwise-star-specific":
                    fused = th.cat((experts[curr_modality], avg_modality), 1)  # -> B x 2D
                    temp = self.g_reason_unshared_weights[ii](fused)
                    temp = self.g_reason_shared(temp)  # B x D -> B x D
                    curr_mask = temp * ind[curr_modality].float().to(dev).unsqueeze(1)

                elif self.use_ce == "pairwise-star-tensor":
                    mod0_reduce = self. dim_reduce(experts[curr_modality])
                    mod0_reduce = mod0_reduce .unsqueeze(2)  # B x reduced_dim x1
                    mod1_reduce = self. dim_reduce(avg_modality)
                    mod1_reduce = mod1_reduce .unsqueeze(1)  # B x1 x reduced_dim
                    flat_dim = self.reduce_dim * self.reduce_dim
                    fused = th.matmul(mod0_reduce, mod1_reduce).view(-1, flat_dim)
                    temp = self.g_reason_1(fused)  # B x 2D -> B x D
                    temp = self.g_reason_shared(temp)  # B x D -> B x D
                    curr_mask = temp * ind[curr_modality].float().to(dev).unsqueeze(1)

                elif self.use_ce in {"pairwise", "triplet"}:
                    for modality_pair in all_combinations:
                        mod0, mod1 = modality_pair
                        if self.use_ce == "pairwise":
                            if mod0 == curr_modality:
                                new_key = f"{mod0}_{mod1}"
                                fused = th.cat((experts[mod0], experts[mod1]), 1)
                                temp = self.g_reason_1(fused)  # B x 2D -> B x D
                                temp = self.g_reason_shared(temp)
                                temp_dict[new_key] = temp
                                avail = (ind[mod0].float() * ind[mod1].float())
                                avai_dict[new_key] = avail.to(dev)
                        elif self.use_ce == "triplet":
                            if (curr_modality not in {mod0, mod1}) or self.include_self:
                                new_key = f"{curr_modality}_{mod0}_{mod1}"
                                fused = th.cat((experts[curr_modality], experts[mod0],
                                               experts[mod1]), 1)  # -> B x 2D
                                temp = self.g_reason_1(fused)  # B x 2D -> B x D
                                temp = self.g_reason_shared(temp)
                                temp_dict[new_key] = temp
                                avail = (ind[curr_modality].float() * ind[mod0].float() *
                                         ind[mod1].float()).to(dev)
                                avai_dict[new_key] = avail

                    # Combine the paired features into a mask through elementwise sum
                    for mm in temp_dict:
                        curr_mask += temp_dict[mm] * avai_dict[mm].unsqueeze(1)
                        mask_num += avai_dict[mm]
                    curr_mask = th.div(curr_mask, (mask_num + 0.00000000001).unsqueeze(1))
                else:
                    raise ValueError(f"Unknown CE mechanism: {self.use_ce}")
                curr_mask = self.h_reason(curr_mask)
                experts[curr_modality] = l(experts[curr_modality], curr_mask)

        elif self.concat_mix_experts:
            concatenated = th.cat(tuple(experts.values()), dim=1)
            vid_embd_ = self.video_GU[0](concatenated)
            text_embd_ = text_embd[self.modalities[0]]
            text_embd_ = text_embd_.view(-1, text_embd_.shape[-1])

        elif self.concat_experts:
            vid_embd_ = th.cat(tuple(experts.values()), dim=1)
            text_embd_ = text_embd[self.modalities[0]]
            text_embd_ = text_embd_.view(-1, text_embd_.shape[-1])
        else:
            for modality, layer in zip(self.modalities, self.video_GU):
                experts[modality] = layer(experts[modality])

        if self.training:
            merge_caption_similiarities = "avg"
        else:
            merge_caption_similiarities = self.test_caption_mode

        if self.task == "classification":
            # for modality, layer in zip(self.modalities, self.video_dim_reduce_later):
            #     attempt to perform affordable classifier, might be removed later
            #     experts[modality] = layer(experts[modality])
            concatenated = th.cat(tuple(experts.values()), dim=1)
            preds = self.classifier(concatenated)
            return {"modalities": self.modalities, "class_preds": preds}
        elif self.concat_experts or self.concat_mix_experts:
            # zero pad to accommodate mismatch in sizes (after first setting the number
            # of VLAD clusters for the text to get the two vectors as close as possible
            # in size)
            if text_embd_.shape[1] > vid_embd_.shape[1]:
                sz = (vid_embd_.shape[0], text_embd_.shape[1])
                dtype, device = text_embd_.dtype, text_embd_.device
                vid_embd_padded = th.zeros(size=sz, dtype=dtype, device=device)
                # try:
                #     vid_embd_padded[:, :vid_embd_.shape[1]] = vid_embd_
                # except:
                #     import ipdb; ipdb.set_trace()
                vid_embd_ = vid_embd_padded
            else:
                sz = (text_embd_.shape[0], vid_embd_.shape[1])
                dtype, device = text_embd_.dtype, text_embd_.device
                text_embd_padded = th.zeros(size=sz, dtype=dtype, device=device)
                text_embd_padded[:, :text_embd_.shape[1]] = text_embd_
                text_embd_ = text_embd_padded
            cross_view_conf_matrix = th.matmul(text_embd_, vid_embd_.t())
        elif self.task == "compute_video_embeddings":
            return {"modalities": self.modalities, "embeddings": experts}
        else:
            cross_view_conf_matrix = sharded_cross_view_inner_product(
                ind=ind,
                vid_embds=experts,
                text_embds=text_embd,
                keep_missing_modalities=self.keep_missing_modalities,
                l2renorm=self.l2renorm,
                text_weights=moe_weights,
                subspaces=self.modalities,
                raw_captions=raw_captions,
                merge_caption_similiarities=merge_caption_similiarities,
            )
        return {
            "modalities": self.modalities,
            "cross_view_conf_matrix": cross_view_conf_matrix,
            "text_embds": text_embd,
            "vid_embds": experts,
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


class MimicCEGatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super().__init__()
        self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.cg(x)
        x = F.normalize(x)
        return x


class ReduceDim(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

#         self.fc = nn.Linear(input_dimension, 512)
#         self.fc2 = nn.Linear(512, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        #         x = self.fc2(F.relu(x))
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


class SpatialMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.cg1 = ContextGating(dimension)
        self.cg2 = ContextGating(dimension)

    def forward(self, x):
        x = self.cg1(x)
        return self.cg2(x)


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


class G_reason(nn.Module):
    def __init__(self, same_dim, num_inputs, non_lin):
        super().__init__()
        self.g_reason_1_specific = nn.Linear(same_dim * num_inputs, same_dim)
        self.g_reason_2_specific = nn.Linear(same_dim, same_dim)
        self.non_lin = non_lin

    def forward(self, x):
        x = self.g_reason_1_specific(x)  # B x 2D -> B x D
        x = self.non_lin(x)
        x = self.g_reason_2_specific(x)
        return x


def sharded_cross_view_inner_product(vid_embds, text_embds, text_weights,
                                     subspaces, l2renorm, ind,
                                     keep_missing_modalities,
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
    T, num_caps, _ = text_embds[subspaces[0]].size()
    device = vid_embds[subspaces[0]].device

    # unroll separate captions onto first dimension and treat them separately
    sims = th.zeros(T * num_caps, B, device=device)
    text_weights = text_weights.view(T * num_caps, -1)
    if False:
        mus = [round(x, 3) for x in text_weights.mean(0).detach().cpu().numpy().tolist()]
        stds = [round(x, 3) for x in text_weights.std(0).detach().cpu().numpy().tolist()]
        summary = ">>>"
        for mod, mu, std in zip(subspaces, mus, stds):
            summary += f"{mod}: {mu} +/- {std} "
        print(summary)

    if keep_missing_modalities:
        # assign every expert/text inner product the same weight, even if the expert
        # is missing
        text_weight_tensor = th.ones(T * num_caps, B, len(subspaces),
                                     dtype=text_weights.dtype,
                                     device=text_weights.device)
    else:
        # mark expert availabilities along the second axis
        available = th.ones(1, B, len(subspaces), dtype=text_weights.dtype)
        for ii, modality in enumerate(subspaces):
            available[:, :, ii] = ind[modality]
        available = available.to(text_weights.device)
        msg = "expected `available` modality mask to only contain 0s or 1s"
        assert set(th.unique(available).cpu().numpy()).issubset(set([0, 1])), msg
        # set the text weights along the first axis and combine with availabilities to
        # produce a <T x B x num_experts> tensor
        text_weight_tensor = text_weights.view(T*num_caps, 1, len(subspaces)) * available
        # normalise to account for missing experts
        normalising_weights = text_weight_tensor.sum(2).view(T*num_caps, B, 1)
        text_weight_tensor = th.div(text_weight_tensor, normalising_weights)

    if l2renorm:
        raise NotImplementedError("Do not use renorm until availability fix is complete")
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
        text_embd_ = text_embds[modality].view(T * num_caps, -1)
        msg = "expected weights to be applied to text embeddings"
        assert text_embd_.shape[0] == text_weights.shape[0], msg
        text_embd_ = text_embd_ / l2_mass_text
        weighting = text_weight_tensor[:, :, idx]
        sims += weighting * th.matmul(text_embd_, vid_embd_.t())  # (T x num_caps) x (B)

    if l2renorm:
        # if not (sims.max() < 1 + tol):
        #     import ipdb; ipdb.set_trace()
        assert sims.max() < 1 + tol, "expected cosine similarities to be < 1"
        assert sims.min() > -1 - tol, "expected cosine similarities to be > -1"

    if th.isnan(sims).sum().item():
        import ipdb; ipdb.set_trace()
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
