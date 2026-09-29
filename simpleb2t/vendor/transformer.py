# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer models
"""

import logging
import typing as tp

import torch
from torch import nn
from x_transformers import Encoder  # type: ignore

from .base import BaseModelConfig

logger = logging.getLogger(__name__)


class TransformerEncoderConfig(BaseModelConfig):
    name: tp.Literal["TransformerEncoder"] = "TransformerEncoder"
    heads: int = 8
    depth: int = 12
    attn_dropout: float = 0.1
    ff_dropout: float = 0.0
    use_scalenorm: bool = True
    rotary_pos_emb: bool = True
    use_rmsnorm: bool = False
    residual_attn: bool = False
    scale_residual: bool = True
    resi_dual: bool = True

    def build(self, dim: int) -> nn.Module:
        return TransformerEncoder(dim, config=self)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder model based on x-transformers:
    https://github.com/lucidrains/x-transformers

    """

    def __init__(
        self,
        # Channels
        dim: int,
        config: TransformerEncoderConfig | None = None,
    ):
        super().__init__()
        config = config if config is not None else TransformerEncoderConfig()

        if dim % config.heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by the number of heads ({config.heads})"
            )

        self.transformer_encoder = Encoder(
            dim=dim,
            depth=config.depth,
            heads=config.heads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            use_scalenorm=config.use_scalenorm,
            use_rmsnorm=config.use_rmsnorm,
            rotary_pos_emb=config.rotary_pos_emb,
            residual_attn=config.residual_attn,
            scale_residual=config.scale_residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.transformer_encoder.forward(
            x,
            mask=mask,
        )

        return x
