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
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories


class MLPEncoderModel(MLPModel):
    """MLP-based encoder neural model.

    This model uses a recurrent neural network (RNN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available RNN types are "lstm" and "gru". Observations can be normalized before being passed to
    the RNN. The output of the model can be either deterministic or stochastic, in which case a distribution module is
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
        self.encoder_output_dim = encoder_output_dim
        self.latent_encoder = None

        # resolve encoder observation groups and dimension
        self.encoder_obs_groups, self.encoder_obs_dim = self._get_obs_dim(obs, obs_groups, encoder_obs_set)

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
            self.encoder_obs_normalizer = EmpiricalNormalization(self.encoder_obs_dim)
        else:
            self.encoder_obs_normalizer = torch.nn.Identity()

        # encoder MLP
        self.encoder = MLP(
            input_dim=self.encoder_obs_dim,
            output_dim=encoder_output_dim,
            hidden_dims=encoder_hidden_dims,
            activation=encoder_activation,
        )

    def forward_encoder(
        self,
        obs: TensorDict,
        encoder_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        obs_latent = super().get_latent(obs, masks, hidden_state)
        latent = torch.cat([obs_latent, encoder_state], dim=-1) if encoder_state is not None else obs_latent
        # MLP forward pass
        mlp_output = self.mlp(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the encoder."""
        # get encoder observation
        obs_list = [obs[obs_group] for obs_group in self.encoder_obs_groups]
        latent_encoder = torch.cat(obs_list, dim=-1)
        latent_encoder = self.encoder_obs_normalizer(latent_encoder)
        latent_encoder = self.encoder(latent_encoder)
        self.latent_encoder = latent_encoder

        encoder_norm = torch.norm(self.latent_encoder.clone())
        if encoder_norm > 1e3:
            print("encoder output is too big")
            print(self.latent_encoder)

        # Concatenate proprioceptive observation and normalize
        latent_policy = super().get_latent(obs)

        return torch.cat([latent_policy, latent_encoder], dim=-1)

    def get_encoder_state(self) -> torch.Tensor | None:
        """Return the encoder output (``None`` for MLP)."""
        return self.latent_encoder

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.encoder_output_dim

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _torchMLPEncoderModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPEncoderModel(self, verbose)


class _torchMLPEncoderModel(nn.Module):  # noqa: N801
    """Exportable MLP encoder model for JIT."""

    def __init__(self, model: MLPEncoderModel) -> None:
        """Create a TorchScript-friendly copy of an MLPEncoderModel."""
        super().__init__()
        # Policy-obs branch (from parent MLPModel)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Encoder branch
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        # Shared MLP head
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference.

        Args:
            obs: Pre-concatenated policy observations.
            encoder_obs: Pre-concatenated encoder observations.

        Returns:
            Deterministic action output.
        """
        # Policy latent
        latent_policy = self.obs_normalizer(obs)
        # Encoder latent
        latent_encoder = self.encoder_obs_normalizer(encoder_obs)
        latent_encoder = self.encoder(latent_encoder)
        # Concatenate and run MLP head
        latent = torch.cat([latent_policy, latent_encoder], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP encoder exports)."""
        pass


class _OnnxMLPEncoderModel(nn.Module):
    """Exportable MLP encoder model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPEncoderModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPEncoderModel."""
        super().__init__()
        self.verbose = verbose
        # Policy-obs branch (from parent MLPModel)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_input_size = model.obs_dim
        # Encoder branch
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.encoder_obs_input_size = model.encoder_obs_dim
        # Shared MLP head
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export.

        Args:
            obs: Pre-concatenated policy observations.
            encoder_obs: Pre-concatenated encoder observations.

        Returns:
            Deterministic action output.
        """
        # Policy latent
        latent_policy = self.obs_normalizer(obs)
        # Encoder latent
        latent_encoder = self.encoder_obs_normalizer(encoder_obs)
        latent_encoder = self.encoder(latent_encoder)
        # Concatenate and run MLP head
        latent = torch.cat([latent_policy, latent_encoder], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (
            torch.zeros(1, self.obs_input_size),
            torch.zeros(1, self.encoder_obs_input_size),
        )

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", "encoder_obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
