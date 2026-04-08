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
from rsl_rl.modules import RNN, EmpiricalNormalization, HiddenState


class RNNEncoderModel(MLPModel):
    """RNN-based encoder neural model.

    This model routes two distinct observation streams through separate branches before merging them
    into a shared policy MLP head:

    - **Encoder branch**: a dedicated observation group (``encoder_obs_set``) is normalised and fed
      step-by-step through an RNN (GRU or LSTM). The RNN's output at each step is used as a compact
      temporal latent, capturing history from the encoder observations.
    - **Policy branch**: the remaining observations (``obs_set``) are normalised by the standard
      parent-class normaliser, producing a flat proprioceptive latent.

    The two latents are concatenated and passed to a shared MLP head to produce the final output.

    Architecture diagram::

        encoder_obs ──► [obs normaliser] ──► [RNN] ──► rnn_latent ──┐
                                                                      ├──► cat ──► [MLP] ──► output
        policy_obs  ──► [obs normaliser] ──────────────────────────► ┘

    This is different from :class:`RNNModel`, where *all* observations pass through the RNN.
    It is also different from :class:`MLPEncoderModel`, where a *feedforward* MLP (not an RNN)
    is used as the encoder.
    """

    is_recurrent: bool = True
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
        encoder_obs_normalization: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ) -> None:
        """Initialize the RNN encoder model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for the policy branch (e.g., ``"actor"``).
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the shared policy MLP.
            activation: Activation function of the policy MLP.
            obs_normalization: Whether to normalise the policy observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            encoder_obs_set: Observation set to use for the RNN encoder branch.
            encoder_obs_normalization: Whether to normalise the encoder observations before feeding them to the RNN.
            rnn_type: Type of RNN to use (``"lstm"`` or ``"gru"``).
            rnn_hidden_dim: Dimension of the RNN hidden state (= encoder output dimension).
            rnn_num_layers: Number of stacked RNN layers.
        """
        # Store encoder output dimension before parent __init__ (needed by _get_latent_dim)
        self.encoder_output_dim = rnn_hidden_dim
        self.latent_encoder: torch.Tensor | None = None

        # Resolve encoder observation groups and total input dimension
        self.encoder_obs_groups, self.encoder_obs_dim = self._get_obs_dim(obs, obs_groups, encoder_obs_set)

        # Initialise the parent MLPModel (builds obs_normaliser, mlp, distribution, etc.)
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

        # Encoder observation normaliser
        if encoder_obs_normalization:
            self.encoder_obs_normalizer = EmpiricalNormalization(self.encoder_obs_dim)
        else:
            self.encoder_obs_normalizer = torch.nn.Identity()

        # RNN encoder
        self.rnn = RNN(self.encoder_obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by fusing the RNN encoder output with policy observations.

        Args:
            obs: Observation TensorDict containing both policy and encoder observation groups.
            masks: Padding masks for recurrent batched training (``None`` during rollout).
            hidden_state: Initial RNN hidden state (used during batched training updates).

        Returns:
            Concatenated latent of shape ``(batch, obs_dim + rnn_hidden_dim)``.
        """
        # --- Encoder branch: encoder_obs → normalise → RNN ---
        encoder_obs = torch.cat([obs[g] for g in self.encoder_obs_groups], dim=-1)
        encoder_obs = self.encoder_obs_normalizer(encoder_obs)
        latent_encoder = self.rnn(encoder_obs, masks, hidden_state).squeeze(0)
        self.latent_encoder = latent_encoder

        # --- Policy branch: policy_obs → normalise (parent) ---
        latent_policy = super().get_latent(obs)

        return torch.cat([latent_policy, latent_encoder], dim=-1)

    def get_encoder_state(self) -> torch.Tensor | None:
        """Return the last RNN encoder output."""
        return self.latent_encoder

    # ------------------------------------------------------------------
    # Recurrent state management (delegate to the RNN module)
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the RNN hidden state (all envs, or only those with ``dones == 1``)."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the current RNN hidden state."""
        return self.rnn.hidden_state  # type: ignore[return-value]

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the RNN hidden state for truncated backpropagation through time."""
        self.rnn.detach_hidden_state(dones)

    # ------------------------------------------------------------------
    # Latent dimensionality
    # ------------------------------------------------------------------

    def _get_latent_dim(self) -> int:
        """Return the total latent dimensionality consumed by the shared MLP head."""
        return self.obs_dim + self.encoder_output_dim

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible version of this model."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMEncoderModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUEncoderModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible version of this model."""
        return _OnnxRNNEncoderModel(self, verbose)


# ---------------------------------------------------------------------------
# JIT export wrappers
# ---------------------------------------------------------------------------


class _TorchGRUEncoderModel(nn.Module):
    """Exportable GRU encoder model for JIT (single-step rollout inference)."""

    def __init__(self, model: RNNEncoderModel) -> None:
        """Create a TorchScript-friendly copy of a GRU-based RNNEncoderModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # raw torch module, no wrapper logic
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs: torch.Tensor, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run one GRU encoder step and return the deterministic action output.

        Args:
            obs: Policy observations of shape ``(1, obs_dim)``.
            encoder_obs: Encoder observations of shape ``(1, encoder_obs_dim)``.

        Returns:
            Deterministic action output of shape ``(1, output_dim)``.
        """
        latent_policy = self.obs_normalizer(obs)
        enc = self.encoder_obs_normalizer(encoder_obs)
        enc, h = self.rnn(enc.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore[index]
        latent_encoder = enc.squeeze(0)
        latent = torch.cat([latent_policy, latent_encoder], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset the exported GRU hidden state to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore[index]


class _TorchLSTMEncoderModel(nn.Module):
    """Exportable LSTM encoder model for JIT (single-step rollout inference)."""

    def __init__(self, model: RNNEncoderModel) -> None:
        """Create a TorchScript-friendly copy of an LSTM-based RNNEncoderModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # raw torch module
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs: torch.Tensor, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run one LSTM encoder step and return the deterministic action output.

        Args:
            obs: Policy observations of shape ``(1, obs_dim)``.
            encoder_obs: Encoder observations of shape ``(1, encoder_obs_dim)``.

        Returns:
            Deterministic action output of shape ``(1, output_dim)``.
        """
        latent_policy = self.obs_normalizer(obs)
        enc = self.encoder_obs_normalizer(encoder_obs)
        enc, (h, c) = self.rnn(enc.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore[index]
        self.cell_state[:] = c  # type: ignore[index]
        latent_encoder = enc.squeeze(0)
        latent = torch.cat([latent_policy, latent_encoder], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset the exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore[index]
        self.cell_state[:] = 0.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------


class _OnnxRNNEncoderModel(nn.Module):
    """Exportable RNN encoder model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: RNNEncoderModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an RNNEncoderModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_input_size = model.obs_dim
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.encoder_obs_input_size = model.encoder_obs_dim
        self.rnn = copy.deepcopy(model.rnn.rnn)  # raw torch module
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, encoder_obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export.

        Args:
            obs: Policy observations of shape ``(1, obs_dim)``.
            encoder_obs: Encoder observations of shape ``(1, encoder_obs_dim)``.
            h_in: Initial RNN hidden state of shape ``(num_layers, 1, rnn_hidden_dim)``.
            c_in: Initial LSTM cell state (``None`` for GRU).

        Returns:
            Tuple of ``(actions, h_out, c_out)`` where ``c_out`` is ``None`` for GRU.
        """
        latent_policy = self.obs_normalizer(obs)
        enc = self.encoder_obs_normalizer(encoder_obs)
        if self.rnn_type == "lstm":
            enc, (h, c) = self.rnn(enc.unsqueeze(0), (h_in, c_in))
            latent_encoder = enc.squeeze(0)
            out = self.deterministic_output(self.mlp(torch.cat([latent_policy, latent_encoder], dim=-1)))
            return out, h, c
        else:
            enc, h = self.rnn(enc.unsqueeze(0), h_in)
            latent_encoder = enc.squeeze(0)
            out = self.deterministic_output(self.mlp(torch.cat([latent_policy, latent_encoder], dim=-1)))
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.obs_input_size)
        enc_obs = torch.zeros(1, self.encoder_obs_input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, enc_obs, h_in, c_in)
        return (obs, enc_obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "encoder_obs", "h_in", "c_in"]
        return ["obs", "encoder_obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
