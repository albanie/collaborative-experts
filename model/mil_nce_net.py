from typing import Dict, Tuple

import torch
from typeguard import typechecked

from base import BaseModel


class MNNet(BaseModel):

    @typechecked
    def __init__(
            self,
            text_dim: int,
            expert_dims: Dict[str, Tuple[int, int]],
            **_unused,
    ):
        self.text_dim = text_dim
        self.expert_dims = expert_dims
        self.modalities = list(expert_dims.keys())
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.ones(1) * 1E-5)

    @typechecked
    def forward(
            self,
            text: torch.Tensor,
            ind: Dict[str, torch.Tensor],
            experts: Dict[str, torch.Tensor],
            **_unused,
    ):
        self.sanity_checks(text=text, experts=experts, ind=ind)
        vid_embedding = next(iter(experts.values()))
        vid_embedding = self.dummy_param + vid_embedding
        text = text.view(text.shape[0] * text.shape[1], text.shape[-1])
        # text = text / torch.norm(text, p=2, dim=1).reshape(-1, 1)
        # vid_embedding = vid_embedding / torch.norm(vid_embedding, p=2,
        #                                            dim=1).reshape(-1, 1)
        sims = torch.matmul(text, vid_embedding.t())
        return {
            "modalities": self.modalities,
            "cross_view_conf_matrix": sims,
            "text_embds": {self.modalities[0]: text},
            "vid_embds": {self.modalities[0]: vid_embedding},
        }

    @typechecked
    def sanity_checks(
            self,
            text: torch.Tensor,
            ind: Dict[str, torch.Tensor],
            experts: Dict[str, torch.Tensor],
    ):
        msg = f"Text dim {text.shape[-1]} did not match expected {self.text_dim}"
        assert text.shape[-1] == self.text_dim, msg
        assert len(experts) == 1, "Expected single modality experts"
        assert len(text.shape) == 4, "Expected four axes for text input"
        assert text.shape[2] == 1, "Expected singleton for text input on dim 2"
        for expert in self.expert_dims:
            msg = f"Expected all features to be present for {expert}"
            assert ind[expert].sum() == len(ind[expert]), msg
            feats = experts[expert]
            expected = self.expert_dims[expert]
            msg = f"Feature shape {feats.shape[1]} did not match expected {expected}"
            assert feats.shape[1] == expected[-1], msg
