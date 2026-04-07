# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for L2T and RecurrentL2T algorithms."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import L2T, RecurrentL2T
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 8
CRITIC_OBS_DIM = 10
STUDENT_OBS_DIM = 6
NUM_ACTIONS = 3


def make_l2t_obs(num_envs: int = NUM_ENVS, device: str = "cpu") -> TensorDict:
    """Create observations with dedicated critic and student keys."""
    return TensorDict(
        {
            "critic": torch.randn(num_envs, CRITIC_OBS_DIM, device=device),
            "student": torch.randn(num_envs, STUDENT_OBS_DIM, device=device),
        },
        batch_size=[num_envs],
        device=device,
    )


def _make_teacher(obs: TensorDict, obs_groups: dict[str, list[str]], recurrent: bool = False) -> MLPModel:
    kwargs: dict[str, object] = {
        "hidden_dims": [32, 32],
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    if recurrent:
        kwargs.update({"rnn_type": "gru", "rnn_hidden_dim": 16, "rnn_num_layers": 1})
        return RNNModel(obs, obs_groups, "critic", NUM_ACTIONS, **kwargs)
    return MLPModel(obs, obs_groups, "critic", NUM_ACTIONS, **kwargs)


def _make_critic(obs: TensorDict, obs_groups: dict[str, list[str]], recurrent: bool = False) -> MLPModel:
    kwargs: dict[str, object] = {"hidden_dims": [32, 32]}
    if recurrent:
        kwargs.update({"rnn_type": "gru", "rnn_hidden_dim": 16, "rnn_num_layers": 1})
        return RNNModel(obs, obs_groups, "critic", 1, **kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **kwargs)


def _make_student(obs: TensorDict, obs_groups: dict[str, list[str]], recurrent: bool = False) -> MLPModel:
    kwargs: dict[str, object] = {
        "hidden_dims": [32, 32],
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    if recurrent:
        kwargs.update({"rnn_type": "gru", "rnn_hidden_dim": 16, "rnn_num_layers": 1})
        return RNNModel(obs, obs_groups, "student", NUM_ACTIONS, **kwargs)
    return MLPModel(obs, obs_groups, "student", NUM_ACTIONS, **kwargs)


def _collect_steps(alg: L2T, obs: TensorDict, num_steps: int = NUM_STEPS) -> None:
    for _ in range(num_steps):
        _ = alg.act(obs)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(obs, rewards, dones, extras={"time_outs": torch.zeros(NUM_ENVS)})


def test_l2t_updates_teacher_and_student_parameters() -> None:
    """L2T update should change both teacher and student parameters."""
    obs = make_l2t_obs()
    obs_groups = {"critic": ["critic"], "student": ["student"]}

    teacher = _make_teacher(obs, obs_groups, recurrent=False)
    critic = _make_critic(obs, obs_groups, recurrent=False)
    student = _make_student(obs, obs_groups, recurrent=False)

    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = L2T(
        teacher,
        critic,
        student,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        schedule="fixed",
        learning_rate=1e-3,
    )

    teacher_before = {k: v.clone() for k, v in alg.teacher.state_dict().items()}
    student_before = {k: v.clone() for k, v in alg.student.state_dict().items()}

    _collect_steps(alg, obs)
    alg.compute_returns(obs)
    losses = alg.update()

    teacher_changed = any(not torch.equal(teacher_before[k], v) for k, v in alg.teacher.state_dict().items())
    student_changed = any(not torch.equal(student_before[k], v) for k, v in alg.student.state_dict().items())

    assert teacher_changed, "Teacher parameters should change after update"
    assert student_changed, "Student parameters should change after update"
    assert "student" in losses and "surrogate" in losses


def test_recurrent_l2t_update_runs() -> None:
    """RecurrentL2T should support recurrent mini-batch updates without errors."""
    obs = make_l2t_obs()
    obs_groups = {"critic": ["critic"], "student": ["student"]}

    teacher = _make_teacher(obs, obs_groups, recurrent=True)
    critic = _make_critic(obs, obs_groups, recurrent=True)
    student = _make_student(obs, obs_groups, recurrent=True)

    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = RecurrentL2T(
        teacher,
        critic,
        student,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",
        learning_rate=1e-3,
    )

    _collect_steps(alg, obs)
    alg.compute_returns(obs)
    losses = alg.update()

    assert "student_imitation" in losses
    assert torch.isfinite(torch.tensor(losses["student_imitation"]))
