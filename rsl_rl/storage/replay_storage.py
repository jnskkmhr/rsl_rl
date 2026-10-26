# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded vector replay, adapted from FlashSAC's vector_replay.py at 11a276c."""

from __future__ import annotations

import math
import torch
from collections import deque
from tensordict import TensorDict


class ReplayStorage:
    """Own replay tensors and n-step staging, ending each row at its first episode boundary."""

    def __init__(
        self,
        example: TensorDict,
        capacity: int,
        max_bytes: int,
        gamma: float,
        n_step: int = 3,
        sum_keys: tuple[str, ...] = ("rewards",),
    ) -> None:
        """Allocate a ring buffer under an explicit storage and staging memory budget."""
        self.num_envs = example.batch_size[0]
        if capacity < self.num_envs or n_step < 1 or max_bytes < 1 or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("Replay requires a vector-sized capacity, positive budgets and a discount in (0, 1].")
        required = {"observations", "next_observations", "actions", "rewards", "terminated", "truncated"}
        if not required <= set(example.keys()):
            raise ValueError("Replay requires observations, successors, actions, rewards and both boundary flags.")
        if any(
            example[key].dtype != torch.bool or example[key].shape != (self.num_envs,)
            for key in ("terminated", "truncated")
        ):
            raise ValueError("Replay boundary flags must be boolean vectors.")
        if example["rewards"].shape != (self.num_envs,):
            raise ValueError("Replay rewards must be a vector.")
        if "rewards" not in sum_keys or len(set(sum_keys)) != len(sum_keys):
            raise ValueError("sum_keys must include rewards exactly once.")
        for key in sum_keys:
            if key not in example or not example[key].is_floating_point():
                raise ValueError(f"Replay accumulated field {key} must be floating point.")
        self.device = example["actions"].device
        self.capacity = capacity
        self.gamma = gamma
        self.n_step = n_step
        self.sum_keys = sum_keys
        self.input_spec = {key: (value.shape, value.dtype) for key, value in example.items(True, True)}
        for key, value in example["observations"].items(True, True):
            other = example["next_observations"][key]
            if other.shape != value.shape or other.dtype != value.dtype:
                raise ValueError("Replay observation and successor schemas must match.")
        template = example.clone(recurse=False)
        template["bootstrap_discount"] = torch.empty(self.num_envs, device=self.device)
        template["bootstrap_steps"] = torch.empty(self.num_envs, device=self.device, dtype=torch.int32)
        input_bytes = sum(t.numel() * t.element_size() for t in example.values(True, True))
        output_bytes = sum(t.numel() * t.element_size() for t in template.values(True, True))
        self.allocated_bytes = capacity * (output_bytes // self.num_envs)
        self.staging_reserved_bytes = n_step * input_bytes + output_bytes
        if self.allocated_bytes + self.staging_reserved_bytes > max_bytes:
            raise ValueError("Replay and n-step staging exceed the configured memory budget.")
        self.storage = TensorDict({}, batch_size=[capacity], device=self.device)
        for key, value in template.items(True, True):
            self.storage[key] = torch.empty((capacity, *value.shape[1:]), dtype=value.dtype, device=self.device)
        self.pending = deque()
        self.position = 0
        self.size = 0

    @property
    def pending_transitions(self) -> int:
        """Collected transitions awaiting sufficient successor steps."""
        return len(self.pending) * self.num_envs

    def drop_pending(self) -> int:
        """Discard incomplete starts when the simulator restarts; never invent terminal transitions."""
        count = self.pending_transitions
        self.pending.clear()
        return count

    @torch.no_grad()
    def add(self, transition: TensorDict) -> None:
        """Copy one vector transition and emit the oldest n-step start when ready."""
        spec = {key: (value.shape, value.dtype) for key, value in transition.items(True, True)}
        if spec != self.input_spec or any(t.device != self.device for t in transition.values(True, True)):
            raise ValueError("Replay transition shape, dtype, keys or device changed.")
        self.pending.append(transition.detach().clone())
        if len(self.pending) < self.n_step:
            return
        result = self.pending[0].clone(recurse=False)
        for key in (*self.sum_keys, "next_observations", "terminated", "truncated"):
            result[key] = self.pending[-1][key]
        steps = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        for earlier in reversed(list(self.pending)[:-1]):
            done = earlier["terminated"] | earlier["truncated"]
            for key in self.sum_keys:
                mask = done.reshape(-1, *([1] * (result[key].ndim - 1)))
                result[key] = earlier[key] + self.gamma * torch.where(mask, 0.0, result[key])
            result["next_observations"] = earlier["next_observations"].where(done, result["next_observations"])
            for key in ("terminated", "truncated"):
                result[key] = torch.where(done, earlier[key], result[key])
            steps = torch.where(done, 1, steps + 1)
        result["bootstrap_steps"] = steps
        result["bootstrap_discount"] = torch.pow(self.gamma, steps.float())
        indices = (torch.arange(self.num_envs, device=self.device) + self.position) % self.capacity
        self.storage[indices] = result
        self.position = (self.position + self.num_envs) % self.capacity
        self.size = min(self.size + self.num_envs, self.capacity)
        self.pending.popleft()

    def sample(self, batch_size: int) -> TensorDict:
        """Sample owned transitions uniformly with replacement."""
        if self.size == 0 or batch_size < 1:
            raise ValueError("Sampling requires nonempty replay and a positive batch size.")
        return self.storage[torch.randint(self.size, (batch_size,), device=self.device)]
