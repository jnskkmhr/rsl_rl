# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

# from rsl_rl.utils import get_param, resolve_nn_activation


class SelfAttention(nn.Module):
    """Self-attention module.

    Attributes:
        input_dim: Dimension of the input features.
        query: Linear layer for queries.
        key: Linear layer for keys.
        value: Linear layer for values.
        softmax: Softmax layer for attention scores.
    """

    def __init__(self, input_dim: int) -> None:
        """Self-attention module.

        Args:
            input_dim: Dimension of the input features.
        """
        super().__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for the self-attention module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim).
        """
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        if mask is not None:
            scores += mask
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module built from :class:`SelfAttention` heads.

    Each head projects the full ``input_dim`` input down to ``head_dim`` via a
    dedicated linear layer, runs :class:`SelfAttention` on that ``head_dim``
    representation, then all head outputs are concatenated and projected back to
    ``input_dim``.  This gives every head a view of **all** input features (as in
    standard MHA) while fully reusing :class:`SelfAttention` as the per-head
    attention primitive.

    .. code-block:: text

        input [B, L, input_dim]
               │         │            one branch per head
          proj₀(Linear)  projₙ(Linear)   # input_dim → head_dim
               │         │
           SA₀(head_dim) SAₙ(head_dim)   # SelfAttention reused
               └────┬────┘
                cat(dim=-1) → [B, L, input_dim]
                dropout
                out_proj   → [B, L, input_dim]

    Attributes:
        input_dim: Total feature dimension (must be divisible by ``num_heads``).
        num_heads: Number of parallel attention heads.
        head_dim: Per-head feature dimension (``input_dim // num_heads``).
        input_projs: Per-head linear projections ``input_dim → head_dim``.
        heads: Per-head :class:`SelfAttention` instances (each on ``head_dim``).
        out_proj: Output projection ``input_dim → input_dim``.
        dropout: Optional dropout on concatenated head outputs before projection.
    """

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        """Initialize MultiHeadAttention.

        Args:
            input_dim: Dimension of the input features. Must be divisible by ``num_heads``.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability applied to the concatenated head output (0 = disabled).

        Raises:
            ValueError: If ``input_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}).")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Per-head input projections: each maps full input_dim → head_dim
        # so every head attends over all input features
        self.input_projs = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)])
        # Per-head attention, reusing SelfAttention (each operates on head_dim)
        self.heads = nn.ModuleList([SelfAttention(self.head_dim) for _ in range(num_heads)])
        # Output projection: concatenated head outputs → input_dim
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for the multi-head attention module.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_dim)``.
            mask: Optional additive mask of shape ``(batch_size, seq_len, seq_len)``
                passed unchanged to each :class:`SelfAttention` head.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, input_dim)``.
        """
        # Project full input to head_dim for each head independently
        # head_inputs: list of num_heads tensors, each [batch_size, seq_len, head_dim]
        head_inputs = [proj(x) for proj in self.input_projs]

        # Each SelfAttention head attends over its projected head_dim representation
        head_outputs = [head(head_in, mask) for head, head_in in zip(self.heads, head_inputs)]

        # Concatenate head outputs → [batch_size, seq_len, input_dim]
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(out)
        return self.out_proj(out)
