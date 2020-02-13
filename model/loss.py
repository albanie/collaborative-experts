"""This module contains an implementation of the max margin ranking loss, slightly
modified from this code:
https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loss.py

The modification is the `fix_norm` conditional, which removes zero terms from the
diagonal when performing the averaging calculation.

Original licence below.
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

import torch.nn as nn
import torch as th
import torch.nn.functional as F


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = th.nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, x, target):
        return self.loss(x, target)


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = th.nn.CrossEntropyLoss(weight=weight)

    def forward(self, x, target):
        return self.loss(x, target.long().to(x.device))


if __name__ == "__main__":
    loss = BCEWithLogitsLoss()
    x = th.randn(3, requires_grad=True)
    target = th.empty(3).random_(2)
    output = loss(x, target)
    output.backward()
    print(target)
