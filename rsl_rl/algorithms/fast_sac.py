# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Holosoma-style distributional SAC with explicit episode-boundary replay.

The update equations follow jnskkmhr/holosoma_agent at a7fe4c9. Replay remains owned
by OffPolicyRunner: true terminations mask bootstrap, timeouts retain final observations.
"""

from __future__ import annotations

import copy
import math
import torch
from collections.abc import Iterable, Iterator
from tensordict import TensorDict
from torch import nn
from typing import Any

from rsl_rl.algorithms.flash_sac import FlashSAC, _compute_categorical_td_target
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_class, resolve_obs_groups


class FastSAC(FlashSAC):
    """Independent categorical Q targets, mean-Q policy gradients and IID exploration."""

    random_warmup = False
    concatenate_policy_observations = True

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        num_actions: int,
        obs_groups: dict[str, list[str]],
        *,
        observation_shapes: dict[str, tuple[int, ...]],
        obs_normalization: bool = True,
        gamma: float = 0.97,
        target_tau: float = 0.125,
        actor_update_period: int = 4,
        learning_rate: float = 3e-4,
        value_max: float = 20.0,
        temperature_initial: float = 0.001,
        target_entropy_ratio: float = 0.0,
        use_autotune: bool = True,
        weight_decay: float = 0.001,
        max_grad_norm: float = 0.0,
        action_scale: float | list[float] = 1.0,
        use_amp: bool = False,
        amp_dtype: str = "bf16",
        **kwargs: Any,
    ) -> None:
        """Configure Holosoma's optimizer and target semantics without an implicit architecture."""
        if not math.isfinite(weight_decay) or weight_decay < 0 or not math.isfinite(max_grad_norm) or max_grad_norm < 0:
            raise ValueError("Weight decay and gradient clipping must be finite and nonnegative.")
        if not math.isfinite(target_entropy_ratio) or target_entropy_ratio < 0:
            raise ValueError("target_entropy_ratio must be finite and nonnegative.")
        if amp_dtype not in ("bf16", "fp16"):
            raise ValueError("amp_dtype must be bf16 or fp16.")
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_autotune = use_autotune
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        super().__init__(
            actor,
            critic,
            num_actions,
            obs_groups,
            gamma=gamma,
            target_tau=target_tau,
            actor_update_period=actor_update_period,
            learning_rate=learning_rate,
            final_learning_rate=learning_rate,
            value_max=value_max,
            temperature_initial=temperature_initial,
            normalize_reward=False,
            noise_max_repeat=1,
            use_amp=use_amp,
            **kwargs,
        )
        self.target_entropy = -num_actions * target_entropy_ratio
        self.grad_scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp and amp_dtype == "fp16")
        self.action_scale = (
            torch.as_tensor(action_scale, device=self.device, dtype=torch.float32).expand(num_actions).clone()
        )
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action_scale must contain positive finite bounds for every action.")
        self.normalizers = nn.ModuleDict()
        self.observation_widths = {}
        for role in ("actor", "critic"):
            shapes = [observation_shapes[key] for key in obs_groups[role]]
            if any(len(shape) != 1 for shape in shapes):
                raise ValueError("FastSAC observation normalization requires vector observation groups.")
            widths = [shape[0] for shape in shapes]
            self.observation_widths[role] = widths
            self.normalizers[role] = EmpiricalNormalization(sum(widths)) if obs_normalization else nn.Identity()
        self.normalizers.to(self.device)
        self.training_config.update(
            algorithm="FastSAC",
            observation_shapes=observation_shapes,
            obs_normalization=obs_normalization,
            target_entropy_ratio=target_entropy_ratio,
            use_autotune=use_autotune,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            action_scale=action_scale,
            amp_dtype=amp_dtype,
        )

    def make_optimizer(self, name: str, parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        """Use AdamW; Holosoma leaves the temperature's weight decay at PyTorch's 0.01 default."""
        return torch.optim.AdamW(
            parameters,
            lr=self.initial_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01 if name == "temperature" else self.weight_decay,
            fused=self.device.type == "cuda",
        )

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FastSAC:
        """Resolve explicitly supplied models through the existing registry."""
        if cfg.get("torch_compile_mode"):
            raise ValueError("FastSAC currently supports eager training.")
        alg_cfg = copy.deepcopy(cfg["algorithm"])
        alg_cfg.pop("class_name")
        alg_cfg = resolve_symmetry_config(alg_cfg, env)
        groups = resolve_obs_groups(obs, cfg.get("obs_groups", {}), ["actor", "critic"])
        models = {}
        for role in ("actor", "critic"):
            model_class, model_cfg = resolve_class(cfg[role])
            models[role] = model_class(obs, groups, role, env.num_actions, **model_cfg)
        return cls(
            models["actor"],
            models["critic"],
            env.num_actions,
            groups,
            observation_shapes={key: tuple(value.shape[1:]) for key, value in obs.items()},
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )

    @torch.no_grad()
    def _normalize_observations(self, obs: TensorDict, role: str, update: bool) -> TensorDict:
        """Normalize each branch separately, excluding critic actions and auxiliary labels."""
        values = torch.cat([obs[key] for key in self.obs_groups[role]], dim=-1)
        normalizer = self.normalizers[role]
        if update and isinstance(normalizer, EmpiricalNormalization):
            normalizer.update(values)
        normalized = normalizer(values)
        result = obs.clone(recurse=False)
        for key, value in zip(self.obs_groups[role], normalized.split(self.observation_widths[role], dim=-1)):
            result[key] = value
        return result

    def sample_policy(
        self, obs: torch.Tensor | TensorDict, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample an IID action and include the per-action scale Jacobian."""
        actions, info = self.actor(obs, training=training)
        scale = self.action_scale * self.action_support
        return actions * scale, dict(info, log_prob=info["log_prob"] - scale.log().sum())

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect from the stochastic policy, including before the first update."""
        normalized = self._normalize_observations(obs, "actor", update=False)
        return self.sample_policy(self.actor_observations(normalized), training=False)[0]

    def deterministic_actions(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate the normalized actor without changing observation statistics."""
        normalized = self._normalize_observations(obs, "actor", update=False)
        mean, _ = self.actor.get_mean_and_std(self.actor_observations(normalized), training=False)
        return mean.tanh() * self.action_scale * self.action_support

    def _step(self, name: str, loss: torch.Tensor) -> None:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.reduce_parameters(optimizer)
        self.grad_scaler.unscale_(optimizer)
        if self.max_grad_norm > 0 and name != "temperature":
            params = [p for group in optimizer.param_groups for p in group["params"]]
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        self.schedulers[name].step()

    def _prepare_batch(self, batch: TensorDict) -> TensorDict:
        prepared = batch.clone(recurse=False)
        for role in ("actor", "critic"):
            prepared[f"{role}_obs"] = self._normalize_observations(batch["observations"], role, update=True)
            prepared[f"next_{role}_obs"] = self._normalize_observations(batch["next_observations"], role, update=True)
        return prepared

    def update_batches(self, batches: Iterable[TensorDict]) -> Iterator[dict[str, torch.Tensor]]:
        """Update statistics once per collected batch, keeping each minibatch's symmetry pairs together."""
        batches = list(batches)
        sizes = [batch.batch_size[0] for batch in batches]
        if self.symmetry:
            batches = [self.symmetry.augment_replay_batch(batch) for batch in batches]
        prepared = self._prepare_batch(torch.cat(batches, dim=0))
        offset = 0
        for i, (batch, size) in enumerate(zip(batches, sizes)):
            count = batch.batch_size[0]
            # Holosoma updates the policy at i=1,5,...; frequency=1 must update every time.
            actor_due = i % self.actor_update_period == (1 if self.actor_update_period > 1 else 0)
            if len(batches) == 1:
                actor_due = self.update_step % self.actor_update_period == 0
            yield self._update(prepared[offset : offset + count], size, actor_due)
            offset += count

    def update(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Run one update, using the continuous policy-update clock for direct callers."""
        return next(self.update_batches([batch]))

    def _update(self, batch: TensorDict, original_size: int, actor_due: bool) -> dict[str, torch.Tensor]:
        actor_obs = self.actor_observations(batch["actor_obs"])
        next_actor_obs = self.actor_observations(batch["next_actor_obs"])
        critic_obs = self.critic_observations(batch["critic_obs"])
        next_critic_obs = self.critic_observations(batch["next_critic_obs"])
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_actions, info = self.sample_policy(next_actor_obs, training=False)
                next_log_prob = info["log_prob"]
                _, target_info = self.target_critic(next_critic_obs, next_actions, training=False)
                targets = torch.stack([
                    _compute_categorical_td_target(
                        log_probs,
                        batch["rewards"],
                        batch["terminated"].float(),
                        self.temperature() * next_log_prob,
                        batch["bootstrap_discount"].reshape(-1, 1),
                        self.num_bins,
                        -self.value_max,
                        self.value_max,
                    )
                    for log_probs in target_info["log_prob"]
                ])
            _, info = self.critic(critic_obs, batch["actions"], training=True)
            critic_loss = -(targets * info["log_prob"]).sum(-1).mean(dim=1).sum()
            critic_loss = critic_loss + self.auxiliary_loss("critic", batch)
        self._step("critic", critic_loss)
        metrics = {"critic/loss": critic_loss}
        if self.use_autotune:
            temperature_loss = -(self.temperature() * (next_log_prob.detach() + self.target_entropy)).mean()
            self._step("temperature", temperature_loss)
            metrics["temperature/loss"] = temperature_loss
        metrics["temperature/value"] = self.temperature()
        if actor_due:
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                actions, info = self.sample_policy(actor_obs, training=True)
                flags = [p.requires_grad for p in self.critic.parameters()]
                self.critic.requires_grad_(False)
                try:
                    values, _ = self.critic(critic_obs, actions, training=False)
                finally:
                    for p, flag in zip(self.critic.parameters(), flags):
                        p.requires_grad_(flag)
                actor_loss = (self.temperature().detach() * info["log_prob"] - values.mean(dim=0)).mean()
                actor_loss = actor_loss + self.auxiliary_loss("actor", batch)
                if self.symmetry:
                    mirror_loss = self.symmetry.compute_policy_loss(
                        self.deterministic_actions,
                        batch["observations"],
                        original_size,
                        augmented=self.symmetry.use_data_augmentation,
                    )
                    if self.symmetry.use_mirror_loss:
                        actor_loss = actor_loss + self.symmetry.mirror_loss_coeff * mirror_loss
                    metrics["symmetry"] = mirror_loss
            self._step("actor", actor_loss)
            metrics.update({"actor/loss": actor_loss, "actor/entropy": -info["log_prob"].mean()})
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target_critic.parameters()), list(self.critic.parameters()), self.target_tau)
        self.update_step += 1
        metrics.update(self.update_auxiliary(batch))
        if self.is_multi_gpu:
            self.synchronize_buffers()
        return {key: value.detach() for key, value in metrics.items()}

    def save(self) -> dict:
        """Save learner state and the separate observation statistics."""
        state = super().save()
        state["fast_sac_version"] = 1
        state["observation_normalizers"] = self.normalizers.state_dict()
        return state

    def load(self, state: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore compatible FastSAC weights, optimizer state and normalization."""
        if state.get("fast_sac_version") != 1:
            raise ValueError("FastSAC requires a FastSAC checkpoint; FlashSAC checkpoints need explicit conversion.")
        result = super().load(state, load_cfg, strict)
        flags = load_cfg or {"actor": True, "critic": True}
        for role in ("actor", "critic"):
            if flags.get(role):
                prefix = role + "."
                self.normalizers[role].load_state_dict(
                    {
                        key[len(prefix) :]: value
                        for key, value in state["observation_normalizers"].items()
                        if key.startswith(prefix)
                    },
                    strict=strict,
                )
        return result

    def train_mode(self) -> None:
        """Enable learning and observation-statistics updates."""
        super().train_mode()
        self.normalizers.train()

    def eval_mode(self) -> None:
        """Freeze observation statistics during evaluation."""
        super().eval_mode()
        self.normalizers.eval()

    def get_policy(self) -> nn.Module:
        """Build a deterministic actor with its normalization and action bounds."""
        return _FastSACPolicy(self)


class _FastSACPolicy(nn.Module):
    """Export only the actor, its observation statistics and action scaling."""

    def __init__(self, algorithm: FastSAC) -> None:
        super().__init__()
        actor = algorithm.actor
        self.actor = actor.as_inference() if callable(getattr(actor, "as_inference", None)) else copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(algorithm.normalizers["actor"])
        self.obs_groups = algorithm.obs_groups["actor"]
        self.widths = algorithm.observation_widths["actor"]
        self.concatenate = algorithm.concatenate_policy_observations
        self.register_buffer("action_scale", algorithm.action_scale * algorithm.action_support)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        normalized = self.normalizer(torch.cat([obs[key] for key in self.obs_groups], dim=-1))
        if self.concatenate:
            inputs = normalized
        else:
            inputs = obs.clone(recurse=False)
            for key, value in zip(self.obs_groups, normalized.split(self.widths, dim=-1)):
                inputs[key] = value
        mean, _ = self.actor.get_mean_and_std(inputs, training=False)
        return mean.tanh() * self.action_scale

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass
