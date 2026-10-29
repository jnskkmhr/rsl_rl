# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules.mlp import EnsembleMLP
from rsl_rl.utils import resolve_class, unpad_trajectories


class MLPModel(nn.Module):
    """MLP-based neural model.

    This model uses a simple multi-layer perceptron (MLP) to process 1D observation groups. Observations can be
    normalized before being passed to the MLP. The output of the model can be either deterministic or
    stochastic, in which case a distribution module is used to sample the outputs.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

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
        extra_input_dim: int = 0,
        layer_norm: bool = False,
        num_ensembles: int = 1,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            extra_input_dim: Width of an additional runtime input, such as a Q critic's action.
            layer_norm: Apply LayerNorm before each hidden activation.
            num_ensembles: Number of independently parameterized MLPs.
        """
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if extra_input_dim < 0:
            raise ValueError("extra_input_dim must be nonnegative.")
        self.extra_input_dim = extra_input_dim

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class, dist_cfg = resolve_class(distribution_cfg)
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP
        if num_ensembles < 1:
            raise ValueError("num_ensembles must be positive.")
        args = (self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)
        self.mlp = (
            MLP(*args, layer_norm=layer_norm)
            if num_ensembles == 1
            else EnsembleMLP(num_ensembles, *args, layer_norm=layer_norm)
        )

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            for mlp in self.mlp.members if isinstance(self.mlp, EnsembleMLP) else [self.mlp]:
                self.distribution.init_mlp_weights(mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        extra_input: torch.Tensor | None = None,
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
        if extra_input is not None:
            extra_input = unpad_trajectories(extra_input, masks) if masks is not None else extra_input
            latent = self.get_latent(obs, masks, hidden_state, extra_input=extra_input)
        else:
            latent = self.get_latent(obs, masks, hidden_state)
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
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        extra_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        # Normalize observations
        latent = self.obs_normalizer(latent)
        if self.extra_input_dim:
            if extra_input is None or extra_input.shape[-1] != self.extra_input_dim:
                raise ValueError("The model requires extra_input with its configured width.")
            latent = torch.cat((latent, extra_input), dim=-1)
        elif extra_input is not None:
            raise ValueError("This model has no configured extra input.")
        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach therecurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Select and concatenate observations
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            # Update the normalizer parameters
            self.obs_normalizer.update(mlp_obs)  # type: ignore

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.extra_input_dim


class _TorchMLPModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: MLPModel) -> None:
        """Create a TorchScript-friendly copy of an MLPModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
