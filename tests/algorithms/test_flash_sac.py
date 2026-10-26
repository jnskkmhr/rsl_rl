# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""FlashSAC target, replay, checkpoint and runner contracts."""

import copy
import torch
from pathlib import Path
from tensordict import TensorDict
from types import SimpleNamespace
from typing import ClassVar

import pytest

from rsl_rl.algorithms import FlashSAC
from rsl_rl.algorithms.flash_sac import _compute_categorical_td_target
from rsl_rl.runners import OffPolicyRunner
from rsl_rl.storage import ReplayStorage
from tests.extensions.test_symmetry import _negate_augmentation


class LinearPolicy(torch.nn.Module):
    """Minimal test policy proving that core FlashSAC needs no reference architecture."""

    def __init__(self, obs: TensorDict, groups: dict, role: str, num_actions: int) -> None:
        """Map selected observations to Gaussian means."""
        super().__init__()
        self.linear = torch.nn.Linear(sum(obs[key].shape[-1] for key in groups[role]), num_actions)
        self.log_std = torch.nn.Parameter(torch.zeros(num_actions))

    def get_mean_and_std(self, obs: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose pre-tanh parameters for the algorithm's exploration rule."""
        return self.linear(obs), self.log_std.exp().expand(obs.shape[0], -1)

    def forward(self, obs: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Keep the path from the sampled action to policy parameters differentiable."""
        mean, std = self.get_mean_and_std(obs, training)
        normal = torch.distributions.Normal(mean, std)
        latent = normal.rsample()
        actions = latent.tanh()
        correction = 2 * (torch.log(torch.tensor(2.0)) - latent - torch.nn.functional.softplus(-2 * latent))
        return actions, {"log_prob": (normal.log_prob(latent) - correction).sum(-1)}


class LinearCategoricalQ(torch.nn.Module):
    """Minimal action-conditioned test critic with no residual blocks or normalization."""

    def __init__(self, obs: TensorDict, groups: dict, role: str, num_actions: int, num_bins: int = 11) -> None:
        """Construct independent linear Q logits."""
        super().__init__()
        self.num_bins = num_bins
        self.linear = torch.nn.Linear(sum(obs[key].shape[-1] for key in groups[role]) + num_actions, 2 * num_bins)
        self.register_buffer("value_support", torch.linspace(-5.0, 5.0, num_bins))

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return expected Q values and categorical log probabilities."""
        logits = self.linear(torch.cat((obs, actions), dim=-1)).reshape(-1, 2, self.num_bins).transpose(0, 1)
        log_prob = logits.log_softmax(-1)
        return (log_prob.exp() * self.value_support).sum(-1), {"log_prob": log_prob}


def config() -> dict:
    """Configure independent linear test heads and bounded replay for contract tests."""
    return dict(
        algorithm=dict(class_name="FlashSAC", num_bins=11, normalize_reward=False),
        actor=dict(class_name=LinearPolicy),
        critic=dict(class_name=LinearCategoricalQ),
        obs_groups=dict(actor=["policy"], critic=["policy"]),
        num_steps_per_env=1,
        updates_per_step=2,
        batch_size=4,
        replay_capacity=32,
        replay_max_bytes=1_000_000,
        warmup_transitions=4,
        n_step=3,
        save_interval=10,
    )


def transition(value: float = 0.0) -> TensorDict:
    """Create a two-environment transition with distinct rewards."""
    obs = TensorDict({"policy": torch.full((2, 3), value)}, [2])
    return TensorDict(
        dict(
            observations=obs,
            next_observations=obs + 1,
            actions=torch.zeros(2, 1),
            rewards=torch.tensor([1.0, 10.0]),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
        ),
        [2],
    )


def test_categorical_target_masks_death_but_bootstraps_timeout() -> None:
    # Deterministic successor value +1. The first row is dead; the second is a timeout.
    """Check categorical projection against hand-calculated terminal and timeout targets."""
    probs = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    target = _compute_categorical_td_target(
        probs.log(),
        torch.zeros(2),
        torch.tensor([1.0, 0.0]),
        torch.zeros(2),
        torch.tensor([[0.5], [0.25]]),
        3,
        -1.0,
        1.0,
    )
    torch.testing.assert_close(target, torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.75, 0.25]]))


def test_replay_owns_samples_and_stops_each_row_at_first_boundary() -> None:
    """Protect replay ownership, boundary handling and ring-buffer eviction."""
    first = transition()
    first["truncated"][0] = True
    first["next_observations", "policy"][0] = 7
    replay = ReplayStorage(first, capacity=4, max_bytes=10000, gamma=0.5)
    replay.add(first)
    first["next_observations", "policy"].zero_()  # Simulated borrowed environment storage.
    second = transition(1.0)
    second["terminated"][1] = True
    second["truncated"][1] = True
    replay.add(second)
    replay.add(transition(2.0))
    rows = replay.storage[:2]
    torch.testing.assert_close(rows["rewards"], torch.tensor([1.0, 15.0]))
    torch.testing.assert_close(rows["bootstrap_discount"], torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(rows["next_observations", "policy"], torch.tensor([[7.0, 7.0, 7.0], [2.0, 2.0, 2.0]]))
    assert rows["terminated"].tolist() == [False, True]
    for step in range(3, 8):
        replay.add(transition(float(step)))
    assert replay.size == 4
    assert set(replay.sample(100)["observations", "policy"][:, 0].tolist()) == {4.0, 5.0}
    assert replay.drop_pending() == 4
    with pytest.raises(ValueError, match="memory budget"):
        ReplayStorage(first, 4, 10, 0.5)


def test_return_normalization_includes_terminal_reward_without_episode_leak() -> None:
    """Include terminal rewards in statistics without carrying them into the next episode."""
    obs = transition()["observations"]
    cfg = config()
    cfg["algorithm"]["normalize_reward"] = True
    alg = FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    alg.observe_rewards(torch.tensor([2.0]), torch.tensor([True]), torch.tensor([False]))
    alg.observe_rewards(torch.tensor([3.0]), torch.tensor([False]), torch.tensor([False]))
    torch.testing.assert_close(alg.reward_mean, torch.tensor([2.5]))
    torch.testing.assert_close(alg.reward_return, torch.tensor([3.0]))


def test_learner_checkpoint_restores_next_update() -> None:
    """Restore the same subsequent update, including optimizer, target and RNG state."""
    obs = transition()["observations"]
    alg = FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), config(), "cpu")
    batch = transition()
    batch["bootstrap_discount"] = torch.full((2,), 0.99)
    alg.update(batch)
    state = copy.deepcopy(alg.save())
    expected_metrics = alg.update(batch)
    expected = copy.deepcopy(alg.save())
    restored = FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), config(), "cpu")
    restored.load(state)
    actual_metrics = restored.update(batch)
    for key in expected_metrics:
        torch.testing.assert_close(actual_metrics[key], expected_metrics[key])
    for name, model in restored.models.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected["models"][name][key])


class AutoResetEnv:
    """Small deterministic VecEnv whose terminal and reset observations differ."""

    num_envs = 2
    num_actions = 1
    device = "cpu"
    cfg: ClassVar[dict] = {}
    max_episode_length = 3

    def __init__(self) -> None:
        """Initialize independent trajectory counters and observations."""
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.obs = TensorDict({"policy": torch.zeros(2, 3)}, [2])

    def get_observations(self) -> TensorDict:
        """Expose the environment-owned observation storage."""
        return self.obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance and reset episodes, preserving the pre-reset observation."""
        self.episode_length_buf += 1
        self.obs["policy"] += 1
        final = self.obs.clone()
        done = self.episode_length_buf == 3
        terminated = done & torch.tensor([True, False])
        truncated = done & torch.tensor([False, True])
        self.obs["policy"][done] = -5
        self.episode_length_buf[done] = 0
        return (
            self.obs,
            -actions.square().sum(-1),
            done,
            dict(terminated=terminated, truncated=truncated, final_obs=final, final_obs_mask=done),
        )


def test_runner_uses_terminal_successors_and_exports_without_replay(tmp_path: Path) -> None:
    """Exercise the runner boundary and verify exported actor outputs at a different batch size."""
    env = AutoResetEnv()
    runner = OffPolicyRunner(env, config())
    runner.learn(6)
    assert runner.alg.update_step == 6
    assert runner.collection_steps == 6
    assert not (runner.replay.storage[:2]["next_observations", "policy"] == -5).any()
    runner.save(str(tmp_path / "model.pt"))
    other = OffPolicyRunner(AutoResetEnv(), config())
    with pytest.warns(UserWarning, match="fresh replay"):
        other.load(str(tmp_path / "model.pt"))
    assert other.replay is None
    policy = other.get_inference_policy()
    expected = policy(env.get_observations())
    runner.export_policy_to_jit(str(tmp_path))
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(exported(env.obs["policy"]), expected)
    runner.export_policy_to_onnx(str(tmp_path))
    import onnxruntime

    session = onnxruntime.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    output = session.run(None, {"obs": env.obs["policy"].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(output), expected, atol=1e-5, rtol=1e-5)


def test_core_requires_explicit_models_and_preserves_policy_gradient() -> None:
    """Reject implicit architectures and retain the gradient through Q to sampled actions."""
    obs = transition()["observations"]
    cfg = config()
    missing = copy.deepcopy(cfg)
    missing["actor"] = {}
    with pytest.raises(ValueError, match=r"explicit actor\.class_name"):
        FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), missing, "cpu")
    alg = FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    actions, _ = alg.sample_policy(alg.actor_observations(obs), training=True)
    q, _ = alg.critic(alg.critic_observations(obs), actions, training=False)
    gradient = torch.autograd.grad(q.mean(), alg.actor.linear.bias)[0]
    assert gradient.abs().sum() > 0
    bad = LinearCategoricalQ(obs, cfg["obs_groups"], "critic", 1)
    bad.value_support.add_(1)
    with pytest.raises(ValueError, match="support must match"):
        FlashSAC(alg.actor, bad, 1, cfg["obs_groups"], num_bins=11)


def test_replay_augmentation_matches_explicit_paired_update() -> None:
    """Automatic augmentation must give the same update as explicitly paired replay rows."""
    obs = transition()["observations"]
    cfg = config()
    augmented_cfg = copy.deepcopy(cfg)
    augmented_cfg["algorithm"]["symmetry_cfg"] = {
        "data_augmentation_func": _negate_augmentation,
        "use_data_augmentation": True,
    }
    env = SimpleNamespace(num_actions=1)
    torch.manual_seed(13)
    manual = FlashSAC.construct_algorithm(obs, env, cfg, "cpu")
    torch.manual_seed(13)
    automatic = FlashSAC.construct_algorithm(obs, env, augmented_cfg, "cpu")
    batch = transition(1.0)
    batch["bootstrap_discount"] = torch.tensor([0.99, 0.99**2])
    reflected = batch.clone()
    for key in ("observations", "next_observations", "actions"):
        reflected[key] = -reflected[key]
    paired = torch.cat((batch, reflected), dim=0)
    torch.manual_seed(23)
    expected = manual.update(paired)
    torch.manual_seed(23)
    actual = automatic.update(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    for name, model in automatic.models.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, manual.models[name].state_dict()[key])
    assert "env" not in automatic.save()["training_config"]["symmetry_cfg"]
    assert automatic.reward_count.item() == 0


def test_mirror_regularization_changes_actor_only_on_scheduled_updates() -> None:
    """Mirror loss contributes to actor optimization even when replay augmentation is disabled."""
    obs = transition()["observations"]
    cfg = config()
    cfg["algorithm"]["symmetry_cfg"] = {
        "data_augmentation_func": _negate_augmentation,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.7,
    }
    alg = FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    with torch.no_grad():
        alg.actor.linear.weight.zero_()
        alg.actor.linear.bias.fill_(0.5)
    batch = transition(1.0)
    batch["bootstrap_discount"] = torch.full((2,), 0.99)
    baseline_cfg = config()
    baseline = FlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), baseline_cfg, "cpu")
    for name, model in alg.models.items():
        baseline.models[name].load_state_dict(model.state_dict())
    expected_symmetry = (2 * torch.tanh(torch.tensor(0.5))).square()
    torch.manual_seed(31)
    expected = baseline.update(batch)
    torch.manual_seed(31)
    actual = alg.update(batch)
    torch.testing.assert_close(actual["symmetry"], expected_symmetry)
    torch.testing.assert_close(actual["actor/loss"], expected["actor/loss"] + 0.7 * expected_symmetry)
    weights = copy.deepcopy(alg.actor.state_dict())
    assert "symmetry" not in alg.update(batch)
    for key, value in alg.actor.state_dict().items():
        torch.testing.assert_close(value, weights[key])


@pytest.mark.parametrize(
    "kind,params,expected",
    [
        ("linear", {"start_step": 2, "end_step": 4, "start_value": 1.0, "end_value": 3.0}, [1, 1, 1, 2, 3, 3]),
        ("discrete", {"steps": [0, 2, 4], "values": [1, 2, 4]}, [1, 1, 2, 2, 4, 4]),
    ],
)
def test_parameter_scheduler_boundaries(kind: str, params: dict, expected: list) -> None:
    """Hold/ramp endpoints and milestone inclusivity are independent of any learning parameter."""
    from rsl_rl.utils import ParameterScheduler

    scheduler = ParameterScheduler(type=kind, params=params)
    assert [scheduler(step) for step in range(6)] == expected
    # Evaluation has no hidden progress: arbitrary access/resume yields the same value.
    assert scheduler(2) == expected[2]


@pytest.mark.parametrize(
    "schedule",
    [
        {"type": "unknown", "params": {}},
        {"type": "linear", "params": {"start_step": 1, "end_step": 1, "start_value": 1, "end_value": 2}},
        {"type": "discrete", "params": {"steps": [1, 2], "values": [1, 2]}},
        {"type": "discrete", "params": {"steps": [0, 2], "values": [1, float("nan")]}},
    ],
)
def test_parameter_scheduler_rejects_ambiguous_or_nonfinite_schedules(schedule: dict) -> None:
    """Malformed schedules fail at construction before collection starts."""
    from rsl_rl.utils import ParameterScheduler

    with pytest.raises(ValueError):
        ParameterScheduler(**schedule)


def test_curriculum_changes_runtime_parameters_and_overrides_cosine_lr() -> None:
    """Schedules affect optimizer groups, policy density and derived temperature targets."""
    import math

    cfg = config()
    cfg["algorithm"]["curriculum_cfg"] = {
        "learning_rate": {
            "type": "linear",
            "params": {
                "start_step": 0,
                "end_step": 4,
                "start_value": 3e-4,
                "end_value": 1e-4,
            },
        },
        "action_support": {
            "type": "linear",
            "params": {
                "start_step": 0,
                "end_step": 4,
                "start_value": 1.0,
                "end_value": 3.0,
            },
        },
        "target_tau": {"type": "discrete", "params": {"steps": [0, 2], "values": [0.01, 0.02]}},
        "temperature_target_sigma": {"type": "discrete", "params": {"steps": [0, 2], "values": [0.15, 0.2]}},
        "actor_update_period": {"type": "discrete", "params": {"steps": [0, 2], "values": [2, 1]}},
    }
    env = AutoResetEnv()
    alg = FlashSAC.construct_algorithm(env.obs, env, cfg, "cpu")
    alg.update_curriculum(2)
    assert alg.action_support == 2 and alg.target_tau == 0.02 and alg.actor_update_period == 1
    assert alg.target_entropy == pytest.approx(0.5 * math.log(2 * math.pi * math.e * 0.2**2))
    obs = alg.actor_observations(env.obs)
    torch.manual_seed(37)
    raw, raw_info = alg.actor(obs, training=False)
    torch.manual_seed(37)
    actions, info = alg.sample_policy(obs, training=False)
    torch.testing.assert_close(actions, raw * 2)
    torch.testing.assert_close(info["log_prob"], raw_info["log_prob"] - math.log(2))
    for name in alg.optimizers:
        loss = sum(p.square().sum() for p in alg.models[name].parameters())
        alg._step(name, loss)
        assert alg.optimizers[name].param_groups[0]["lr"] == pytest.approx(2e-4)
    state = copy.deepcopy(alg.save())
    restored = FlashSAC.construct_algorithm(env.obs, env, cfg, "cpu")
    restored.load(state)
    assert restored.curriculum_step == 2 and restored.action_support == 2
    assert restored.target_tau == 0.02 and restored.target_entropy == alg.target_entropy
    assert restored.learning_rate == pytest.approx(2e-4)


def test_curriculum_runner_clock_resume_and_replay_coordinates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Warmup counts vector steps, saved support is frozen for inference, and old actions are not rescaled."""
    cfg = config()
    cfg["warmup_transitions"] = 32
    cfg["algorithm"]["curriculum_cfg"] = {
        "action_support": {
            "type": "linear",
            "params": {
                "start_step": 2,
                "end_step": 4,
                "start_value": 1.0,
                "end_value": 3.0,
            },
        }
    }
    monkeypatch.setattr(torch, "rand", lambda *size, **kwargs: torch.ones(*size, **kwargs))
    runner = OffPolicyRunner(AutoResetEnv(), cfg)
    runner.learn(3)
    assert runner.collection_steps == 3 and runner.alg.curriculum_step == 2
    assert runner.alg.action_support == 1
    old_actions = runner.replay.storage[:2]["actions"].clone()
    runner.save(str(tmp_path / "curriculum.pt"))
    restored = OffPolicyRunner(AutoResetEnv(), cfg)
    with pytest.warns(UserWarning, match="fresh replay"):
        restored.load(str(tmp_path / "curriculum.pt"))
    assert restored.get_inference_policy().action_support == 1
    restored.learn(1)
    assert restored.collection_steps == 4 and restored.alg.curriculum_step == 3
    assert restored.alg.action_support == 2
    runner.learn(2)
    assert runner.alg.action_support == 3
    torch.testing.assert_close(runner.replay.storage[:2]["actions"], old_actions)


@pytest.mark.parametrize("name", ["gamma", "num_bins", "typo"])
def test_curriculum_rejects_parameters_without_runtime_support(name: str) -> None:
    """Changing replay discount or model dimensions cannot be implemented by assigning an attribute."""
    cfg = config()
    cfg["algorithm"]["curriculum_cfg"] = {name: {"type": "discrete", "params": {"steps": [0], "values": [1]}}}
    env = AutoResetEnv()
    with pytest.raises(ValueError, match="not runtime-schedulable"):
        FlashSAC.construct_algorithm(env.obs, env, cfg, "cpu")


def test_runner_relabels_accumulated_components_with_current_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay retains unweighted n-step costs and applies the current curriculum at sampling."""

    class RewardTransform:
        sum_keys = ("rewards", "cost")

        def __init__(self, env: AutoResetEnv) -> None:
            self.weight = 1.0

        def start(self, remaining_seconds: float | None) -> None:
            pass

        def before_collection(self, step: int) -> None:
            self.weight = 1.0 if step < 4 else 3.0

        def process_transition(self, batch: TensorDict) -> TensorDict:
            batch["cost"] = torch.ones_like(batch["rewards"])
            return batch

        def process_batch(self, batch: TensorDict) -> TensorDict:
            # An independent geometric sum verifies accumulation stops at episode boundaries.
            expected_cost = (1 - batch["bootstrap_discount"]) / (1 - 0.99)
            torch.testing.assert_close(batch["cost"], expected_cost, atol=3e-6, rtol=3e-6)
            batch["rewards"] = batch["rewards"] - self.weight * batch["cost"]
            batch["applied_weight"] = torch.full_like(batch["rewards"], self.weight)
            return batch

    cfg = config()
    cfg["replay_cfg"] = {"class_name": RewardTransform}
    runner = OffPolicyRunner(AutoResetEnv(), cfg)
    weights = []
    update = runner.alg.update

    def record_update(batch: TensorDict) -> dict[str, torch.Tensor]:
        weights.append(batch["applied_weight"][0].item())
        return update(batch)

    monkeypatch.setattr(runner.alg, "update", record_update)
    runner.learn(6)
    assert weights == [1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    assert "applied_weight" not in runner.replay.storage


def test_runner_deadline_stops_updates_and_checkpoints_count_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline reached inside a collection's updates still saves the final learner."""
    import rsl_rl.runners.off_policy_runner as runner_module

    clock = [100.0]
    monkeypatch.setattr(runner_module.time, "perf_counter", lambda: clock[0])
    env = AutoResetEnv()
    step = env.step

    def timed_step(actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        clock[0] += 0.1
        return step(actions)

    monkeypatch.setattr(env, "step", timed_step)
    cfg = config()
    cfg.update(n_step=1, warmup_transitions=2, max_time_s=1.0, save_interval_updates=2, logger="tensorboard")
    runner = OffPolicyRunner(env, cfg, log_dir=str(tmp_path))
    update = runner.alg.update

    def timed_update(batch: TensorDict) -> dict[str, torch.Tensor]:
        result = update(batch)
        clock[0] += 0.3
        return result

    monkeypatch.setattr(runner.alg, "update", timed_update)
    runner.learn(100, process_started=100.0)
    assert runner.collection_steps == 2
    assert runner.alg.update_step == 3
    assert (tmp_path / "model_update_2.pt").is_file()
    assert (tmp_path / "model_2.pt").is_file()
