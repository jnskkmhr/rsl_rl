# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_ae_model import MLPAutoEncoderModel
from rsl_rl.models.mlp_encoder_model import MLPEncoderModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories


class MLPVAEModel(MLPAutoEncoderModel):
    """MLP-based VAE encoder-decoder model.

    Extends :class:`MLPAutoEncoderModel` by making the encoder variational.
    The encoder MLP outputs ``(mu, log_var)`` of shape ``(batch, encoder_output_dim)``
    each. A decoder reconstructs the encoder observations from the sampled latent ``z``.

    During training ``z`` is drawn via the reparameterization trick so that gradients
    flow through both the reconstruction loss and the KL divergence. During eval /
    inference ``mu`` is returned directly, giving a deterministic and stable embedding
    that is suitable for student distillation.

    Architecture diagram::

        encoder_obs ──► [normalizer] ──► [Encoder MLP] ──► (mu, log_var)
                                                                  │
                                            reparameterize ──► z ──► [Decoder MLP] ──► reconstructed
                                                                  │
                        policy_obs ──────────── cat(policy_latent, mu) ──► [Policy MLP] ──► output

    Note:
        ``mu`` (not ``z``) is stored as the encoder state in the rollout buffer and
        used by the policy MLP during both rollout collection and the PPO update step.
        This keeps rollout behaviour consistent with inference and provides a stable
        target for the student encoder to imitate during distillation.
    """

    has_decoder = True

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
        decoder_obs_set: str = "decoder",
    ) -> None:
        """Initialize the MLP VAE model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set used by the policy MLP (e.g. ``"actor"``).
            output_dim: Dimension of the policy output (number of actions).
            hidden_dims: Hidden dimensions of the policy MLP.
            activation: Activation function for the policy MLP.
            obs_normalization: Whether to normalize policy observations.
            distribution_cfg: Configuration dictionary for the output distribution.
            encoder_obs_set: Observation set fed to the encoder / decoder branches.
            encoder_output_dim: Bottleneck dimensionality ``d_z`` (= dim of ``mu``, ``log_var``, and ``z``).
            encoder_hidden_dims: Hidden dimensions of the encoder MLP. The decoder uses
                these reversed, i.e. ``encoder_hidden_dims[::-1]``.
            encoder_activation: Activation function for both encoder and decoder MLPs.
            encoder_obs_normalization: Whether to normalize encoder observations.
            decoder_obs_set: Observation set used as the decoder reconstruction target.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            encoder_obs_set,
            encoder_output_dim,
            encoder_hidden_dims,
            encoder_activation,
            encoder_obs_normalization,
            decoder_obs_set,
        )
        # Replace the deterministic encoder (output_dim = encoder_output_dim) created by the
        # parent with a variational one that outputs (mu, log_var): 2 * encoder_output_dim.
        # The decoder and policy MLP input dim are unchanged (both use encoder_output_dim).
        self.encoder = MLP(
            input_dim=self.encoder_obs_dim,
            output_dim=2 * encoder_output_dim,
            hidden_dims=list(encoder_hidden_dims),
            activation=encoder_activation,
        )

        # VAE statistics from the most recent forward pass (set by get_latent / forward_encoder)
        self.mu: torch.Tensor | None = None
        self.log_var: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the VAE encoder, returning ``(mu, log_var)``."""
        encoder_obs = torch.cat([obs[g] for g in self.encoder_obs_groups], dim=-1)
        encoder_obs = self.encoder_obs_normalizer(encoder_obs)
        out = self.encoder(encoder_obs)  # (batch, 2 * encoder_output_dim)
        return out.chunk(2, dim=-1)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample ``z`` via the reparameterization trick during training; return ``mu`` during eval."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent using ``mu`` for the policy (no sampling during rollout).

        ``mu`` is consistent with inference behaviour and is stored in the rollout buffer
        via :meth:`get_encoder_state`. VAE sampling is deferred to :meth:`forward_encoder`
        so the decoder gradient always sees a sampled ``z``.
        """
        mu, log_var = self._encode(obs)
        self.mu = mu
        self.log_var = log_var
        # Store mu as the deterministic encoder state for the rollout buffer
        self.latent_encoder = mu

        # Policy-obs latent — calls MLPModel.get_latent, bypassing encoder logic
        latent_policy = super(MLPEncoderModel, self).get_latent(obs, masks, hidden_state)
        return torch.cat([latent_policy, mu], dim=-1)

    def forward_encoder(
        self,
        obs: TensorDict,
        encoder_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the PPO update step.

        Re-runs the VAE encoder on the current batch to obtain fresh ``(mu, log_var)``,
        samples ``z`` for the decoder reconstruction, and caches both for the KL loss.
        The policy MLP uses the ``encoder_state`` cached in the rollout buffer (= ``mu``
        from rollout collection) following the standard PPO off-policy trick.
        """
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        # Fresh VAE encoder pass — updates mu, log_var, and decoder_output
        mu, log_var = self._encode(obs)
        self.mu = mu
        self.log_var = log_var
        z = self._reparameterize(mu, log_var)
        self.decoder_output = self.decoder(z)

        # Policy MLP: use cached encoder_state from the rollout buffer (off-policy)
        latent_policy = super(MLPEncoderModel, self).get_latent(obs, masks, hidden_state)
        latent = torch.cat([latent_policy, encoder_state], dim=-1) if encoder_state is not None else latent_policy
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_vae_params(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return ``(mu, log_var)`` from the most recent :meth:`forward_encoder` call.

        Returns ``None`` before the first forward pass. Intended for use by
        :class:`~rsl_rl.algorithms.PPOVAE` to compute the KL divergence loss.
        """
        if self.mu is None or self.log_var is None:
            return None
        return self.mu, self.log_var

    def get_decoder_inference(self, latent_encoder: torch.Tensor) -> torch.Tensor:
        """Run the decoder on a provided latent directly (no sampling)."""
        return self.decoder(latent_encoder)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible export wrapper."""
        return _torchMLPVAEModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible export wrapper."""
        return _OnnxMLPVAEModel(self, verbose)

    def as_jit_decoder(self) -> nn.Module:
        """Return a TorchScript-compatible decoder-only export wrapper."""
        return _torchMLPVAEDecoderModel(self)

    def as_onnx_decoder(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible decoder-only export wrapper."""
        return _OnnxMLPVAEDecoderModel(self, verbose)


class _torchMLPVAEModel(nn.Module):  # noqa: N801
    """TorchScript-exportable wrapper for MLPVAEModel.

    Uses ``mu`` (no sampling) for deterministic inference, matching the behaviour
    of :meth:`MLPVAEModel.get_latent` in eval mode.
    """

    def __init__(self, model: MLPVAEModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)  # outputs 2 * encoder_output_dim
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
        latent_policy = self.obs_normalizer(obs)
        enc_out = self.encoder_obs_normalizer(encoder_obs)
        enc_out = self.encoder(enc_out)  # (batch, 2 * encoder_output_dim)
        mu, _ = enc_out.chunk(2, dim=-1)  # use mu, discard log_var
        latent = torch.cat([latent_policy, mu], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP VAE exports)."""
        pass


class _OnnxMLPVAEModel(nn.Module):
    """ONNX-exportable wrapper for MLPVAEModel.

    Uses ``mu`` (no sampling) for deterministic inference, matching the behaviour
    of :meth:`MLPVAEModel.get_latent` in eval mode.
    """

    is_recurrent: bool = False

    def __init__(self, model: MLPVAEModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_input_size = model.obs_dim
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)  # outputs 2 * encoder_output_dim
        self.encoder_obs_input_size = model.encoder_obs_dim
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
        latent_policy = self.obs_normalizer(obs)
        enc_out = self.encoder_obs_normalizer(encoder_obs)
        enc_out = self.encoder(enc_out)  # (batch, 2 * encoder_output_dim)
        mu, _ = enc_out.chunk(2, dim=-1)  # use mu, discard log_var
        latent = torch.cat([latent_policy, mu], dim=-1)
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


class _torchMLPVAEDecoderModel(nn.Module):  # noqa: N801
    """TorchScript-exportable decoder-only wrapper for MLPVAEModel.

    Takes the encoder output (``mu``) directly and returns the decoder reconstruction,
    without running the encoder or policy MLP.
    """

    def __init__(self, model: MLPVAEModel) -> None:
        super().__init__()
        self.decoder = copy.deepcopy(model.decoder)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Run the decoder on a pre-computed latent.

        Args:
            encoder_output: Encoder latent (``mu``) of shape ``(batch, encoder_output_dim)``.

        Returns:
            Decoded reconstruction of shape ``(batch, decoder_obs_dim)``.
        """
        return self.decoder(encoder_output)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for decoder exports)."""
        pass


class _OnnxMLPVAEDecoderModel(nn.Module):
    """ONNX-exportable decoder-only wrapper for MLPVAEModel.

    Takes the encoder output (``mu``) directly and returns the decoder reconstruction,
    without running the encoder or policy MLP.
    """

    is_recurrent: bool = False

    def __init__(self, model: MLPVAEModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.decoder = copy.deepcopy(model.decoder)
        self.encoder_output_dim = model.encoder_output_dim

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Run the decoder on a pre-computed latent for ONNX export.

        Args:
            encoder_output: Encoder latent (``mu``) of shape ``(batch, encoder_output_dim)``.

        Returns:
            Decoded reconstruction of shape ``(batch, decoder_obs_dim)``.
        """
        return self.decoder(encoder_output)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return a representative dummy input for ONNX tracing."""
        return (torch.zeros(1, self.encoder_output_dim),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["encoder_output"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["decoded"]
