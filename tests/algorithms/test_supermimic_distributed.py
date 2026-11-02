# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distributed SuperMimic synchronization and failure handling."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

from rsl_rl.algorithms import SuperMimic
from tests.algorithms.test_supermimic import ScalarEnv, _cfg, _collect


def worker(rank: int, rendezvous: str) -> None:
    """Use distinct local samples while maintaining one shared learner."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42 + rank)
        env = ScalarEnv()
        env.position.fill_(rank)
        cfg = _cfg()
        cfg["multi_gpu"] = dict(world_size=2, global_rank=rank, local_rank=rank)
        learner = SuperMimic.construct_algorithm(env.get_observations(), env, cfg, "cpu")
        learner.broadcast_parameters()
        _collect(learner, env, 4)
        metrics = learner.update()
        assert metrics and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        for model in (learner.actor, learner.world_model, learner.critic):
            for param in model.state_dict().values():
                reference = param.clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(param, reference, atol=0, rtol=0)
        # One nonfinite gradient must skip the same optimizer step on every rank.
        model = learner.actor
        before = [p.detach().clone() for p in model.parameters()]
        loss = sum(p.sum() for p in model.parameters()) * (float("nan") if rank else 1.0)
        learner._apply(loss, "actor", model)
        for p, old in zip(model.parameters(), before):
            torch.testing.assert_close(p, old, atol=0, rtol=0)
        # A rank can have no valid dynamics samples; moments still use the global batch.
        norm = learner.world_model.in_norm
        width = norm._mean.shape[-1]
        norm.update(torch.ones(0 if rank else 3, width))
        assert torch.isfinite(norm._mean).all() and torch.isfinite(norm._var).all()
    finally:
        dist.destroy_process_group()


def test_two_rank_update(tmp_path: Path) -> None:
    """Independent replay must produce synchronized parameters and optimizer decisions."""
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
