# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under CC BY-NC 4.0.
# See LICENSE in this directory. Modifications for SimpleB2T are described
# in THIRD_PARTY_NOTICES.md at the repository root.

"""Pydantic configurations for models."""

import pydantic
import torch.nn as nn


class BaseModelConfig(pydantic.BaseModel):
    """Base class for model configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError
