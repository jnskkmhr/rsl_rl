# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Two-process contracts for one shared FlashSAC learner."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.runners import OffPolicyRunner
from tests.algorithms.test_flash_sac import AutoResetEnv, config


class RankEnv(AutoResetEnv):
    """Give ranks different rewards and n-step flush times."""

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Flush rank one immediately and keep distinct local reward distributions."""
        obs, rewards, dones, extras = super().step(actions)
        if dist.get_rank() == 1:
            dones = torch.ones_like(dones)
            extras.update(terminated=dones, truncated=~dones, final_obs=obs.clone(), final_obs_mask=dones)
        return obs, rewards + 3 * dist.get_rank(), dones, extras


def _run_rank(rank: int, rendezvous: str, output: str, device_type: str) -> None:
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    device = f"cuda:{rank}" if device_type == "cuda" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(100 + rank)
        cfg = config()
        cfg["algorithm"]["normalize_reward"] = True
        cfg["algorithm"]["use_amp"] = device_type == "cuda"
        runner = OffPolicyRunner(RankEnv(), cfg, device=device)
        rewards = torch.tensor([1.0, 2.0], device=device) + rank * 10
        runner.alg.observe_rewards(
            rewards, torch.zeros(2, dtype=torch.bool, device=device), torch.zeros(2, dtype=torch.bool, device=device)
        )
        torch.testing.assert_close(runner.alg.reward_mean.cpu(), torch.tensor([6.5]))
        torch.testing.assert_close(runner.alg.reward_var.cpu(), torch.tensor([25.25 + 1e-4 / 4]))
        torch.testing.assert_close(runner.alg.reward_max.cpu(), torch.tensor([12.0]))
        assert runner.alg.reward_count == 4
        runner.learn(6)
        # Each rank must wait for the slowest local n-step replay to warm up.
        assert runner.alg.update_step == 6
        assert runner.collection_steps == 6
        # An analytic optimizer step must include the other rank's gradient.
        parameter = next(runner.alg.temperature.parameters())
        before = parameter.detach().clone()
        runner.alg.optimizers["temperature"] = torch.optim.SGD([parameter], lr=0.1)
        runner.alg._step("temperature", parameter.sum() * (1 + 2 * rank))
        torch.testing.assert_close(parameter, before - 0.2)
        runner.save(str(Path(output) / f"checkpoint-{rank}.pt"))
        torch.save(runner.alg.save(), Path(output) / f"state-{rank}.pt")

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "device_type",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"))],
)
def test_distributed_flash_sac_shares_updates_statistics_and_stopping(tmp_path: Path, device_type: str) -> None:
    """Different rank data must produce one learner, the requested iterations and one checkpoint writer."""
    ctx = mp.get_context("spawn")
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    processes = [
        ctx.Process(target=_run_rank, args=(rank, rendezvous, str(tmp_path), device_type)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=45)
        assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    states = [torch.load(tmp_path / f"state-{rank}.pt", weights_only=False, map_location="cpu") for rank in range(2)]
    for name, model in states[0]["models"].items():
        for key, value in model.items():
            torch.testing.assert_close(value, states[1]["models"][name][key], rtol=0, atol=0)
    for key in ("reward_mean", "reward_var", "reward_max", "reward_count"):
        torch.testing.assert_close(states[0]["statistics"][key], states[1]["statistics"][key])
    assert (tmp_path / "checkpoint-0.pt").exists()
    assert not (tmp_path / "checkpoint-1.pt").exists()
