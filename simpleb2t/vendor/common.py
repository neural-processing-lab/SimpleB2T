# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Common modules to be used with brain models."""

import math
import typing as tp

import torch
from torch import nn
from torchvision.ops import MLP


class ChannelPositions:
    INVALID_VALUE = -0.1


from .base import BaseModelConfig


class BahdanauAttention(nn.Module):
    """Bahdanau attention from [1]_.

    Implementation inspired from pytorch's seq2seq tutorial:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-decoder

    .. [1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by
           jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        if input_size is None:
            self.Wa = nn.LazyLinear(hidden_size)
            self.Ua = nn.LazyLinear(hidden_size)
        else:
            self.Wa = nn.Linear(input_size, hidden_size)
            self.Ua = nn.Linear(input_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, keys, queries=None):
        """
        Parameters
        ----------
        query :
            Query tensor of shape (batch_size, n_features, n_times).
        """
        keys = keys.transpose(2, 1)  # (B, F, T) -> (B, T, F)
        sum_ = self.Wa(keys)
        if queries is not None:
            queries = queries.transpose(2, 1)
            assert queries.shape == keys.shape
            sum_ += self.Ua(queries)

        scores = self.Va(torch.tanh(sum_))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        context = context.transpose(2, 1)  # (B, 1, F) -> (B, F, 1)

        return context


class ChannelDropout(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("See brainmagick.models.common.")

    def forward(self, x):
        raise NotImplementedError


class SubjectLayers(nn.Module):
    """Per subject linear projection.

    Parameters
    ----------
    in_channels :
        Number of input channels.
    out_channels :
        Number of output channels.
    n_subjects :
        Number of subjects to initialize weights for.
    init_id :
        If True, initialize the projection matrices with the identity.
    mode :
        How to apply the linear projection. With "gather" (original implementation), a tensor of
        shape (batch_size, in_channels, out_channels) containing the projection matrices for each
        example in the batch is first created. This tensor can be very large when the number of
        channels is high (e.g. when using on fMRI data with many input voxels). In this case, it
        may be better to use "for_loop": this will loop over each unique subject in the batch to
        apply the projection separately.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
        mode: tp.Literal["gather", "for_loop"] = "gather",
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        if init_id:
            if in_channels != out_channels:
                raise ValueError(
                    "in_channels and out_channels must be the same for identity initialization."
                )
            self.weights.data[:] = torch.eye(in_channels)[None]
        else:
            self.weights.data.normal_()
        self.weights.data *= 1 / in_channels**0.5
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, in_channels, n_times)
        subjects: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:  # (batch_size, out_channels, n_times)
        N, C, D = self.weights.shape
        assert subjects.max() < N, (
            "Subject index higher than number of subjects used to initialize the weights."
        )

        if self.mode == "gather":
            weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
            out = torch.einsum("bct,bcd->bdt", x, weights)
        elif self.mode == "for_loop":
            B, _, T = x.shape
            out = torch.empty((B, D, T), device=x.device)
            for subject in subjects.unique():
                mask = subjects.reshape(-1) == subject
                out[mask] = torch.einsum("bct,cd->bdt", x[mask], self.weights[subject])
        else:
            raise NotImplementedError()

        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """

    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2) ** 0.5
        assert int(n_freqs**2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2) ** 0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat(
            [
                torch.cos(loc),
                torch.sin(loc),
            ],
            dim=-1,
        )
        return emb


class ChannelMerger(nn.Module):
    def __init__(
        self,
        chout: int,
        pos_dim: int = 256,
        dropout: float = 0,
        usage_penalty: float = 0.0,
        n_subjects: int = 200,
        per_subject: bool = False,
    ):
        super().__init__()
        assert pos_dim % 4 == 0
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.invalid_value = ChannelPositions.INVALID_VALUE
        self.heads.data /= pos_dim**0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.0)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, subject_ids, positions, return_weights=False):
        B, C, T = meg.shape  # pylint: disable=unused-variable
        meg = meg.clone()
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        invalid_mask = (positions == self.invalid_value).all(dim=-1)
        score_offset[invalid_mask] = float("-inf")

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float("-inf")

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            heads = self.heads.gather(0, subject_ids.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        if return_weights:
            return weights
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.0:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class UnitNorm(nn.Module):
    """Normalize last dimension of tensor to have unit Frobenius norm.

    Useful for parametrizing different normalization alternatives in `MlpConfig` below.

    NOTE: `hidden_dim` argument included for consistency with other normalization layers (e.g.
          BatchNorm).
    """

    def __init__(self, hidden_dim: int = 0) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(p="fro", dim=-1, keepdim=True)


class MlpConfig(BaseModelConfig):
    """Multilayer perceptron, e.g. for use as projection head.

    Notes
    -----
    Input size can be specified in the config or at build time.
    Output size can either be specified in the config (as the last element of `hidden_sizes`) or at
    build time through the `output_size` parameter (like other models in neuraltrain), in which
    case this will overwrite the last value in `hidden_sizes`.
    For convenience, passing an empty list of hidden sizes yields `nn.Identity`.
    """

    name: tp.Literal["Mlp"] = "Mlp"

    input_size: int | None = None
    hidden_sizes: list[int]

    norm_layer: tp.Literal["layer", "batch", "instance", "unit", None] = None
    activation_layer: tp.Literal["relu", "gelu", "elu", "prelu", None] = "relu"

    bias: bool = True
    dropout: float = 0.0

    @staticmethod
    def _get_norm_layer(kind: str | None) -> tp.Type[nn.Module] | None:
        return {
            "batch": nn.BatchNorm1d,
            "layer": nn.LayerNorm,
            "instance": nn.InstanceNorm1d,
            "unit": UnitNorm,
            None: None,
        }[kind]

    @staticmethod
    def _get_activation_layer(kind: str | None) -> tp.Type[nn.Module]:
        return {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "prelu": nn.PReLU,
            None: nn.Identity,
        }[kind]

    def build(
        self, input_size: int | None = None, output_size: int | None = None
    ) -> nn.Sequential | nn.Identity:
        if not self.hidden_sizes:
            return nn.Identity()

        input_size = self.input_size if input_size is None else input_size
        assert input_size is not None, "input_size cannot be None."
        hidden_sizes = self.hidden_sizes
        if output_size is not None:
            hidden_sizes[-1] = output_size

        return MLP(
            in_channels=input_size,
            hidden_channels=hidden_sizes,
            norm_layer=self._get_norm_layer(self.norm_layer),
            activation_layer=self._get_activation_layer(self.activation_layer),
            bias=self.bias,
            dropout=self.dropout,
        )


class NormDenormScaler(nn.Module):
    """Norm-denorm scaler inspired by [1]_.

    At inference time, this module applies z-score normalization of its input, followed by
    de-normalization based on the statistics of the data seen at instantiation.

    Parameters
    ----------
    x :
        Data on which to fit the denormalizer, of shape (n_examples, n_features).
    affine :
        If True, de-normalize with the statistics of `x`.

    References
    ----------
    .. [1] Ozcelik, Furkan, and Rufin VanRullen. "Natural scene reconstruction from fMRI signals
       using generative latent diffusion." Scientific Reports 13.1 (2023): 15666.
    """

    def __init__(self, x: torch.Tensor, affine: bool = True):
        super().__init__()

        assert x.ndim == 2, "Tensor must be flattened."

        self.scaler = nn.BatchNorm1d(
            x.shape[1], affine=affine, track_running_stats=False, eps=1e-15
        ).eval()

        if affine:
            # Disable gradient as this is not currently intended to be finetuned
            self.scaler.weight.requires_grad = False
            self.scaler.bias.requires_grad = False

            self.scaler.weight.data = x.std(dim=0, correction=0)
            self.scaler.bias.data = x.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler(x)


class Mean(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)
