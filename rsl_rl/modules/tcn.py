# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class CausalConv1dBlock(nn.Module):
    """Single causal Conv1d block with residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: nn.Module = nn.ELU,
    ) -> None:
        """CausalConv1dBlock module.

        Args:
            in_dim: Dimension of the input.
            out_dim: Dimension of the output.
            kernel_size: Kernel size for Conv1d.
            dilation: Dilation factor for Conv1d.
            activation: Activation function.
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation, padding=self.padding)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = activation

        # residual projection if dims differ
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the CausalConv1dBlock module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, out_dim).
        """
        # x: [B, L, in_dim]
        residual = self.residual_proj(x)  # [B, L, out_dim]

        out = x.transpose(1, 2)  # [B, in_dim, L]
        out = self.conv(out)
        out = out[:, :, : -self.padding] if self.padding > 0 else out  # trim future
        out = out.transpose(1, 2)  # [B, L, out_dim]

        out = self.norm(out + residual)
        return self.activation(out)


class TCN(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        kernel_size: int = 3,
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        """TCN module.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers.
            kernel_size: Kernel size for Conv1d.
            activation: Activation function.
            last_activation: Activation function of the last layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        activation_cls = resolve_nn_activation(activation)
        last_activation_cls = resolve_nn_activation(last_activation) if last_activation is not None else None

        # Build TCN blocks with exponential dilation
        blocks = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            # dilation = 2**i  # 1, 2, 4, 8, ...
            dilation = 1
            blocks.append(CausalConv1dBlock(in_dim, h_dim, kernel_size, dilation, activation_cls))
            in_dim = h_dim
        self.blocks = nn.ModuleList(blocks)

        # Final projection layer (no residual, just linear)
        self.output_proj = nn.Linear(in_dim, output_dim)
        self.last_activation = last_activation_cls() if last_activation_cls is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch_size, seq_len, input_dim]

        Returns:
            [batch_size, seq_len, output_dim]
        """
        for block in self.blocks:
            x = block(x)  # [B, L, h_dim]

        x = self.output_proj(x)  # [B, L, output_dim]
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x
