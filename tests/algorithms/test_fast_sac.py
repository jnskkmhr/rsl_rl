# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""FastSAC must retain correct replay boundaries and distributed normalization."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

from rsl_rl.runners import OffPolicyRunner
from rsl_rl.storage import ReplayStorage
from tests.algorithms.test_flash_sac import AutoResetEnv, config, transition


def fast_config() -> dict:
    """Select the Holosoma learner with small replay and vector-step warmup."""
    cfg = config()
    cfg["algorithm"] = dict(class_name="FastSAC", num_bins=11, value_max=5.0)
    cfg.update(n_step=1, learning_starts=0, warmup_transitions=1, replay_steps_per_env=8, batch_size_is_global=True)
    return cfg


def test_replay_wraparound_retains_real_horizons() -> None:
    """Never append stale rewards across the newest transition when replay wraps."""
    replay = ReplayStorage(transition(), capacity=4, max_bytes=100000, gamma=0.9, n_step=3)
    for step in range(1, 9):
        item = transition(float(step))
        item["rewards"].fill_(step)
        replay.add(item)
    rows = replay.storage
    starts = rows["observations", "policy"][:, 0]
    expected = starts + 0.9 * (starts + 1) + 0.9**2 * (starts + 2)
    torch.testing.assert_close(rows["rewards"], expected)
    torch.testing.assert_close(rows["next_observations", "policy"][:, 0], starts + 3)
    assert (rows["bootstrap_steps"] == 3).all()
    assert starts.max() == 6  # Steps 7 and 8 are pending, not padded with old data.


def _rank(rank: int, rendezvous: str, output: str) -> None:
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(10 + rank)
        env = AutoResetEnv()
        env.obs["policy"] += rank * 10
        runner = OffPolicyRunner(env, fast_config(), device="cpu")
        assert runner.cfg["batch_size"] == 2
        runner.learn(6)
        assert runner.alg.update_step == 10
        assert runner.alg.normalizers["actor"].count > 0
        torch.save(runner.alg.save(), Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_fast_sac_distributed_models_and_normalizers_agree(tmp_path: Path) -> None:
    """Independent rank observations must produce identical weights and pooled observation moments."""
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_rank, args=(rank, "file://" + str(tmp_path / "sync"), str(tmp_path)))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=40)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    states = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(2)]
    for role, model in states[0]["models"].items():
        for key, value in model.items():
            torch.testing.assert_close(value, states[1]["models"][role][key], rtol=0, atol=0)
    for key, value in states[0]["observation_normalizers"].items():
        torch.testing.assert_close(value, states[1]["observation_normalizers"][key], rtol=0, atol=0)


def test_observation_variance_includes_between_batch_mean_difference() -> None:
    """Two constant batches at 0 and 2 have pooled variance 1, not 0.25."""
    from rsl_rl.modules import EmpiricalNormalization

    norm = EmpiricalNormalization(1)
    norm.update(torch.zeros(8, 1))
    norm.update(torch.full((8, 1), 2.0))
    torch.testing.assert_close(norm.mean, torch.ones(1))
    torch.testing.assert_close(norm.std, torch.ones(1))
