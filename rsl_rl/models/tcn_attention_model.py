# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import TCN, EmpiricalNormalization, HiddenState, SelfAttention


class TCNAttentionModel(MLPModel):
    """TCN-based encoder neural model.

    This model uses a temporal convolutional network (TCN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available TCN types are "tcn" and "attention". Observations can be normalized before being passed to
    the TCN. The output of the model can be either deterministic or stochastic, in which case a distribution module is
    used to sample the outputs.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""
    has_encoder: bool = True
    """Whether the model contains an encoder."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_obs_set: str = "encoder",
        encoder_output_dim: int = 0,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        encoder_activation: str = "elu",
        encoder_obs_normalization: bool = False,
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            encoder_obs_set: Observation set to use for the encoder.
            encoder_output_dim: Dimension of the encoder output.
            encoder_hidden_dims: Hidden dimensions of the encoder.
            encoder_activation: Activation function of the encoder.
            encoder_obs_normalization: Whether to normalize the observations before feeding them to the encoder.
        """
        # instantiate variables
        self.history_length = obs[encoder_obs_set].shape[1]  # history obs is in [batch, history_length, obs_dim]
        self.encoder_output_dim = encoder_output_dim
        self.latent_encoder = None

        # resolve encoder observation groups and dimension
        self.encoder_obs_groups, self.encoder_obs_dim = self._get_history_obs_dim(obs, obs_groups, encoder_obs_set)

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        if encoder_obs_normalization:
            # TODO: check if empirical normalization works for history data
            self.encoder_obs_normalizer = EmpiricalNormalization(self.encoder_obs_dim)
        else:
            self.encoder_obs_normalizer = torch.nn.Identity()

        self.tcn = TCN(
            input_dim=self.encoder_obs_dim,
            output_dim=encoder_output_dim,
            hidden_dims=encoder_hidden_dims,
            activation=encoder_activation,
        )
        self.attention = SelfAttention(input_dim=encoder_output_dim)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the encoder."""
        # get encoder observation
        obs_list = [obs[obs_group] for obs_group in self.encoder_obs_groups]
        latent_encoder = torch.cat(obs_list, dim=-1)
        latent_encoder = self.encoder_obs_normalizer(latent_encoder)
        latent_encoder = self.tcn(latent_encoder)
        latent_encoder = self.attention(latent_encoder).flatten(start_dim=1)
        self.latent_encoder = latent_encoder

        # Concatenate proprioceptive observation and normalize
        latent_policy = super().get_latent(obs)

        return torch.cat([latent_policy, latent_encoder], dim=-1)

    def get_encoder_state(self) -> torch.Tensor | None:
        """Return the encoder output (``None`` for MLP)."""
        return self.latent_encoder

    def _get_history_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 3:
                raise ValueError(
                    f"The TCN model only supports 3D observations (history), got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.encoder_output_dim * self.history_length

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _torchTCNAttentionModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxTCNAttentionModel(self, verbose)


class _torchTCNAttentionModel(nn.Module):  # noqa: N801
    """Exportable TCN+Attention encoder model for JIT."""

    def __init__(self, model: TCNAttentionModel) -> None:
        """Create a TorchScript-friendly copy of a TCNAttentionModel."""
        super().__init__()
        # Policy-obs branch (from parent MLPModel)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Encoder branch: normalizer → TCN → SelfAttention → flatten
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.tcn = copy.deepcopy(model.tcn)
        self.attention = copy.deepcopy(model.attention)
        # Shared MLP head
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference.

        Args:
            obs: Pre-concatenated policy observations of shape ``(batch, obs_dim)``.
            encoder_obs: History of encoder observations of shape
                ``(batch, history_length, encoder_obs_dim)``.

        Returns:
            Deterministic action output.
        """
        # Policy latent: [batch, obs_dim]
        latent_policy = self.obs_normalizer(obs)
        # Encoder latent: normalize → TCN → attention → flatten
        latent_encoder = self.encoder_obs_normalizer(encoder_obs)  # [batch, L, enc_obs_dim]
        latent_encoder = self.tcn(latent_encoder)  # [batch, L, enc_out_dim]
        latent_encoder = self.attention(latent_encoder)  # [batch, L, enc_out_dim]
        latent_encoder = latent_encoder.flatten(start_dim=1)  # [batch, L * enc_out_dim]
        # Concatenate and run MLP head
        latent = torch.cat([latent_policy, latent_encoder], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for TCN+Attention exports)."""
        pass


class _OnnxTCNAttentionModel(nn.Module):
    """Exportable TCN+Attention encoder model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: TCNAttentionModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a TCNAttentionModel."""
        super().__init__()
        self.verbose = verbose
        # Policy-obs branch (from parent MLPModel)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_input_size = model.obs_dim
        # Encoder branch: normalizer → TCN → SelfAttention → flatten
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.tcn = copy.deepcopy(model.tcn)
        self.attention = copy.deepcopy(model.attention)
        self.history_length = model.history_length
        self.encoder_obs_dim = model.encoder_obs_dim
        # Shared MLP head
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export.

        Args:
            obs: Pre-concatenated policy observations of shape ``(batch, obs_dim)``.
            encoder_obs: History of encoder observations of shape
                ``(batch, history_length, encoder_obs_dim)``.

        Returns:
            Deterministic action output.
        """
        # Policy latent: [batch, obs_dim]
        latent_policy = self.obs_normalizer(obs)
        # Encoder latent: normalize → TCN → attention → flatten
        latent_encoder = self.encoder_obs_normalizer(encoder_obs)  # [batch, L, enc_obs_dim]
        latent_encoder = self.tcn(latent_encoder)  # [batch, L, enc_out_dim]
        latent_encoder = self.attention(latent_encoder)  # [batch, L, enc_out_dim]
        latent_encoder = latent_encoder.flatten(start_dim=1)  # [batch, L * enc_out_dim]
        # Concatenate and run MLP head
        latent = torch.cat([latent_policy, latent_encoder], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (
            torch.zeros(1, self.obs_input_size),
            # encoder_obs is a sequence: [batch, history_length, encoder_obs_dim]
            torch.zeros(1, self.history_length, self.encoder_obs_dim),
        )

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", "encoder_obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
