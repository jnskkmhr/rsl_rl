# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        layer_norm: bool = False,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
            layer_norm: Normalize each hidden layer before its activation.
        """
        super().__init__()

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims_processed:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_mod)
            previous_dim = hidden_dim

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(previous_dim, output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(previous_dim, total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x


class EnsembleMLP(nn.Module):
    """Independent MLPs with batch-first outputs [batch, ensemble, ...]."""

    def __init__(
        self,
        num_ensembles: int,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        layer_norm: bool = False,
    ) -> None:
        """Construct independently initialized ensemble members."""
        super().__init__()
        self.members = nn.ModuleList([
            MLP(input_dim, output_dim, hidden_dims, activation, layer_norm=layer_norm) for _ in range(num_ensembles)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate every member on the same input."""
        return torch.stack([member(x) for member in self.members], dim=1)
