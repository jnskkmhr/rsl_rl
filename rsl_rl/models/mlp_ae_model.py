# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_encoder_model import MLPEncoderModel
from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


class MLPAutoEncoderModel(MLPEncoderModel):
    """MLP-based encoder-decoder neural model.

    Extends :class:`MLPEncoderModel` with a symmetric decoder that reconstructs the encoder
    observations from the encoder latent. The decoder is the mirror image of the encoder:
    its hidden dimensions are the encoder hidden dims reversed, and it outputs the original
    encoder observation dimension.

    Architecture diagram::

        encoder_obs ──► [encoder_obs_normalizer] ──► [Encoder MLP] ──► latent ──► [Decoder MLP] ──► reconstructed
                                                                           │
                                                                           ↓
                         policy_obs ──────────────────► cat(policy_latent, latent) ──► [Policy MLP] ──► output

    The decoder output is stored internally after each forward pass and can be retrieved with
    :meth:`get_decoder_output`. It is intended for use in auxiliary reconstruction losses during
    training (e.g. student encoder distillation or privileged-observation reconstruction).
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
        """Initialize the MLP encoder-decoder model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., ``"actor"`` or ``"critic"``).
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the shared policy MLP.
            activation: Activation function of the policy MLP.
            obs_normalization: Whether to normalize the policy observations.
            distribution_cfg: Configuration dictionary for the output distribution.
            encoder_obs_set: Observation set to use for the encoder / decoder branches.
            encoder_output_dim: Bottleneck dimension of the encoder (= input dim of the decoder).
            encoder_hidden_dims: Hidden dimensions of the encoder MLP. The decoder uses these
                reversed, i.e. ``encoder_hidden_dims[::-1]``.
            encoder_activation: Activation function for both encoder and decoder MLPs.
            encoder_obs_normalization: Whether to normalize the encoder observations.
            decoder_obs_set: Observation set to use for the decoder.
        """
        # resolve decoder observation groups and dimension
        self.decoder_obs_groups, self.decoder_obs_dim = self._get_obs_dim(obs, obs_groups, decoder_obs_set)

        # Initialize parent (builds encoder + policy MLP)
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
        )

        # Decoder: mirror of encoder — reversed hidden dims, outputs encoder_obs_dim
        self.decoder = MLP(
            input_dim=encoder_output_dim,
            output_dim=self.decoder_obs_dim,
            hidden_dims=list(encoder_hidden_dims)[::-1],
            activation=encoder_activation,
        )
        self.decoder_output: torch.Tensor | None = None
        """Most recent decoder reconstruction, populated during :meth:`get_latent`."""

    def forward_encoder(
        self,
        obs: TensorDict,
        encoder_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass that also re-runs the encoder and decoder for the reconstruction loss.

        During ``update()``, this is the entry point (not ``get_latent``). The parent
        implementation uses the cached ``encoder_state`` from the rollout buffer for the policy
        MLP, which is correct for PPO. But the decoder needs a *fresh* encoder output from the
        current batch, so we re-run the encoder here before delegating to the parent MLP logic.
        """
        # Unpad if needed (same as parent)
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        # decode privileged state to compute auxiliary loss
        encoder_obs = torch.cat([obs[g] for g in self.encoder_obs_groups], dim=-1)
        encoder_obs_normalized = self.encoder_obs_normalizer(encoder_obs)
        latent_encoder = self.encoder(encoder_obs_normalized)
        self.decoder_output = self.decoder(latent_encoder)

        # Policy MLP: uses the cached encoder_state from the rollout buffer (PPO off-policy)
        obs_latent = super(MLPEncoderModel, self).get_latent(obs, masks, hidden_state)
        latent = torch.cat([obs_latent, encoder_state], dim=-1) if encoder_state is not None else obs_latent
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_decoder_output(self) -> torch.Tensor | None:
        """Return the most recent decoder reconstruction (``None`` before the first forward pass)."""
        return self.decoder_output

    def get_decoder_inference(self, latent_encoder: torch.Tensor) -> torch.Tensor:
        """Return the decoder reconstruction."""
        return self.decoder(latent_encoder)
