# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .mlp_ae_model import MLPAutoEncoderModel
from .mlp_encoder_model import MLPEncoderModel
from .mlp_model import MLPModel
from .mlp_vae_model import MLPVAEModel
from .rnn_encoder_model import RNNEncoderModel
from .rnn_model import RNNModel
from .tcn_attention_model import TCNAttentionModel
from .tcn_model import TCNModel

__all__ = [
    "CNNModel",
    "MLPAutoEncoderModel",
    "MLPEncoderModel",
    "MLPModel",
    "MLPVAEModel",
    "RNNEncoderModel",
    "RNNModel",
    "TCNAttentionModel",
    "TCNModel",
]
