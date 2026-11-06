# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from rsl_rl.algorithms import SAC, FlashSAC
from rsl_rl.env import VecEnv
from rsl_rl.runners.flash_sac_runner import FlashSACRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import resolve_callable


class OffPolicyRunner(OnPolicyRunner):
    """Off-policy runner for reinforcement learning algorithms.

    This runner uses the interaction, logging, and checkpointing loop provided by :class:`OnPolicyRunner`.
    """

    alg: SAC
    """The SAC algorithm."""

    def __new__(
        cls, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
    ) -> OffPolicyRunner | FlashSACRunner:
        """Preserve the FlashSAC collection loop for existing FlashSAC and FastSAC configs."""
        algorithm = resolve_callable(train_cfg["algorithm"]["class_name"])
        if issubclass(algorithm, FlashSAC):
            return FlashSACRunner(env, train_cfg, log_dir, device)
        return super().__new__(cls)
