# Copyright (c) 2026 Holiday Robotics and NVIDIA CORPORATION
# SPDX-License-Identifier: MIT

"""FlashSAC with external replay and the RSL-RL algorithm construction interface.

Learning computations are ported from FlashSAC 11a276cad45ba8816639f66c210dfc2db755bb2c.
Reward statistics include terminal rewards before clearing ended trajectories, as in Agile.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.distributed as dist
from collections.abc import Iterable, Iterator
from tensordict import TensorDict
from torch import nn

from rsl_rl.env import VecEnv
from rsl_rl.extensions import Symmetry, resolve_symmetry_config
from rsl_rl.utils import ParameterScheduler, reduce_gradients_in_buckets, resolve_class, resolve_obs_groups


@torch.no_grad()
def normalize_parameters(model: nn.Module) -> None:
    """Apply optional model parameter constraints after initialization and optimizer steps."""
    for module in model.modules():
        if hasattr(module, "normalize_parameters"):
            module.normalize_parameters()


class FlashSAC:
    """Distributional SAC over supplied policy and categorical Q models."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        num_actions: int,
        obs_groups: dict[str, list[str]],
        *,
        device: str = "cpu",
        gamma: float = 0.99,
        target_tau: float = 0.01,
        actor_update_period: int = 2,
        learning_rate: float = 3e-4,
        final_learning_rate: float = 1.5e-4,
        decay_updates: int = 100_000,
        num_bins: int = 101,
        value_max: float = 5.0,
        temperature_initial: float = 0.01,
        temperature_target_sigma: float = 0.15,
        normalize_reward: bool = True,
        noise_zeta_exponent: float = 2.0,
        noise_max_repeat: int = 16,
        action_support: float = 1.0,
        use_amp: bool = False,
        symmetry_cfg: dict | None = None,
        curriculum_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
    ) -> None:
        """Construct optimizers and targets; replay allocation belongs to the runner."""
        self.training_config = dict(
            num_actions=num_actions,
            obs_groups=obs_groups,
            gamma=gamma,
            target_tau=target_tau,
            actor_update_period=actor_update_period,
            learning_rate=learning_rate,
            final_learning_rate=final_learning_rate,
            decay_updates=decay_updates,
            num_bins=num_bins,
            value_max=value_max,
            temperature_target_sigma=temperature_target_sigma,
            normalize_reward=normalize_reward,
            noise_zeta_exponent=noise_zeta_exponent,
            noise_max_repeat=noise_max_repeat,
            use_amp=use_amp,
        )
        if curriculum_cfg:
            self.training_config["curriculum_cfg"] = copy.deepcopy(curriculum_cfg)
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None
        if symmetry_cfg:
            self.training_config["symmetry_cfg"] = {key: value for key, value in symmetry_cfg.items() if key != "env"}
        self.device = torch.device(device)
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg else 1
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg else 0
        if not math.isfinite(grad_reduce_bucket_mb) or grad_reduce_bucket_mb <= 0:
            raise ValueError("grad_reduce_bucket_mb must be positive and finite.")
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb
        if use_amp and self.device.type != "cuda":
            raise ValueError("FlashSAC FP16 AMP requires CUDA.")
        positive = (
            learning_rate,
            final_learning_rate,
            decay_updates,
            value_max,
            temperature_initial,
            temperature_target_sigma,
            noise_zeta_exponent,
            noise_max_repeat,
            action_support,
        )
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("FlashSAC rates, dimensions, periods and action support must be positive and finite.")
        if not 0 < gamma <= 1 or not 0 < target_tau <= 1 or num_bins < 2 or actor_update_period < 1:
            raise ValueError("Invalid FlashSAC discount, target update, categorical bins or actor period.")
        self.validate_models(actor, critic, num_bins, value_max)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.temperature = _EntropyTemperature(temperature_initial).to(self.device)
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.gamma = gamma
        self.target_tau = target_tau
        self.actor_update_period = actor_update_period
        self.num_bins = num_bins
        self.value_max = value_max
        self.normalize_reward = normalize_reward
        self.use_amp = use_amp
        self.action_support = action_support
        self.target_entropy = 0.5 * num_actions * math.log(2 * math.pi * math.e * temperature_target_sigma**2)
        self.update_step = 0
        self.optimizers = {}
        self.schedulers = {}
        self.initial_learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_updates = decay_updates
        self.curriculum_step = 0
        self.curriculum = {name: ParameterScheduler(**cfg) for name, cfg in (curriculum_cfg or {}).items()}
        for name, scheduler in self.curriculum.items():
            if name == "actor_update_period" and scheduler.type != "discrete":
                raise ValueError("actor_update_period requires a discrete integer schedule.")
            for value in scheduler.values:
                self._validate_curriculum_value(name, value)
        for name, model in self.models.items():
            if name == "target_critic":
                continue
            parameters = [p for p in model.parameters() if p.requires_grad]
            self.optimizers[name] = self.make_optimizer(name, parameters)
            self.schedulers[name] = torch.optim.lr_scheduler.LambdaLR(self.optimizers[name], self._learning_rate_factor)
        for model in (self.actor, self.critic, self.target_critic):
            normalize_parameters(model)
        self.grad_scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)
        ns = torch.arange(1, noise_max_repeat + 1, dtype=torch.float32, device=self.device)
        pmf = ns ** (-noise_zeta_exponent)
        self.noise_cdf = (pmf / pmf.sum()).cumsum(0)
        self.noise = torch.empty(0, device=self.device)
        self.noise_count = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.noise_repeat = torch.tensor(1, dtype=torch.int32, device=self.device)
        self.reward_return = torch.zeros(1, device=self.device)
        self.reward_max = torch.zeros(1, device=self.device)
        self.reward_mean = torch.zeros(1, device=self.device)
        self.reward_var = torch.ones(1, device=self.device)
        self.reward_count = torch.tensor(0.0, device=self.device)
        self.update_curriculum(0)

    def make_optimizer(self, name: str, parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        """Construct the optimizer for a model's explicitly owned parameters."""
        return torch.optim.Adam(parameters, lr=self.initial_learning_rate, fused=self.device.type == "cuda")

    def update_batches(self, batches: Iterable[TensorDict]) -> Iterator[dict[str, torch.Tensor]]:
        """Update from replay minibatches collected after one vector step."""
        for batch in batches:
            yield self.update(batch)

    @property
    def models(self) -> dict[str, nn.Module]:
        """Named online and target models for checkpoints."""
        return dict(
            actor=self.actor, critic=self.critic, target_critic=self.target_critic, temperature=self.temperature
        )

    @property
    def learning_rate(self) -> float:
        """Current critic learning rate."""
        return self.optimizers["critic"].param_groups[0]["lr"]

    def _learning_rate_factor(self, step: int) -> float:
        if "learning_rate" in self.curriculum:
            return self.curriculum["learning_rate"](self.curriculum_step) / self.initial_learning_rate
        fraction = min(step / self.decay_updates, 1.0)
        rate = self.final_learning_rate + (self.initial_learning_rate - self.final_learning_rate) * 0.5 * (
            1 + math.cos(math.pi * fraction)
        )
        return rate / self.initial_learning_rate

    @staticmethod
    def _validate_curriculum_value(name: str, value: float | int) -> None:
        if name not in (
            "action_support",
            "learning_rate",
            "target_tau",
            "actor_update_period",
            "temperature_target_sigma",
        ):
            raise ValueError(f"FlashSAC parameter {name!r} is not runtime-schedulable.")
        if value <= 0:
            raise ValueError(f"Scheduled {name} must stay positive.")
        if name == "target_tau" and value > 1:
            raise ValueError("Scheduled target_tau must be at most one.")
        if name == "actor_update_period" and type(value) is not int:
            raise ValueError("Scheduled actor_update_period must be an integer.")

    def update_curriculum(self, collection_step: int) -> dict[str, float | int]:
        """Apply schedules using completed vector steps; keep values fixed across associated updates."""
        if type(collection_step) is not int or collection_step < 0:
            raise ValueError("Curriculum progress must be a nonnegative number of completed collection steps.")
        values = {name: scheduler(collection_step) for name, scheduler in self.curriculum.items()}
        self.curriculum_step = collection_step
        for name, value in values.items():
            if name == "learning_rate":
                for optimizer in self.optimizers.values():
                    for group in optimizer.param_groups:
                        group["lr"] = value
            elif name == "temperature_target_sigma":
                self.target_entropy = 0.5 * self.num_actions * math.log(2 * math.pi * math.e * value**2)
            else:
                setattr(self, name, value)
        return values

    @staticmethod
    def validate_models(actor: nn.Module, critic: nn.Module, num_bins: int, value_max: float) -> None:
        """Validate model capabilities without requiring a particular architecture.

        The actor must provide get_mean_and_std and a differentiable forward returning
        (actions, {"log_prob": log_prob}). The critic must expose one-dimensional value_support
        and forward(observations, actions, training), returning Q values [ensemble, batch]
        and categorical log probabilities [ensemble, batch, bins].
        """
        if not callable(getattr(actor, "get_mean_and_std", None)):
            raise TypeError("FlashSAC actor must provide get_mean_and_std and differentiable action sampling.")
        support = getattr(critic, "value_support", None)
        if not isinstance(support, torch.Tensor) or support.shape != (num_bins,):
            raise ValueError("FlashSAC critic must expose value_support with num_bins entries.")
        expected = torch.linspace(-value_max, value_max, num_bins, device=support.device, dtype=support.dtype)
        if not torch.allclose(support, expected):
            raise ValueError("Critic categorical support must match the algorithm configuration.")

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FlashSAC:
        """Construct explicitly configured models through the existing class_name mechanism.

        Model constructors receive (obs, obs_groups, obs_set, num_actions, **model_cfg).
        Their forwards consume concatenated observation tensors; graph integrations can
        override actor_observations and critic_observations to retain named inputs.
        """
        if cfg.get("torch_compile_mode"):
            raise ValueError("FlashSAC currently supports eager training.")
        alg_cfg = copy.deepcopy(cfg["algorithm"])
        alg_cfg.pop("class_name")
        alg_cfg = resolve_symmetry_config(alg_cfg, env)
        groups = resolve_obs_groups(obs, cfg.get("obs_groups", {}), ["actor", "critic"])
        if any(obs[key].ndim != 2 for keys in groups.values() for key in keys):
            raise ValueError("Plain FlashSAC requires vector observation groups.")
        models = {}
        for role in ("actor", "critic"):
            if not cfg.get(role) or "class_name" not in cfg[role]:
                raise ValueError(
                    f"FlashSAC requires an explicit {role}.class_name; no network architecture is implicit."
                )
            model_class, model_cfg = resolve_class(cfg[role])
            models[role] = model_class(obs, groups, role, env.num_actions, **model_cfg)
        return FlashSAC(
            models["actor"],
            models["critic"],
            env.num_actions,
            groups,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )

    def actor_observations(self, obs: TensorDict) -> torch.Tensor:
        """Select actor inputs; compositional algorithms may retain the TensorDict."""
        return torch.cat([obs[key] for key in self.obs_groups["actor"]], dim=-1)

    def critic_observations(self, obs: TensorDict) -> torch.Tensor:
        """Select critic inputs independently of actor inputs."""
        return torch.cat([obs[key] for key in self.obs_groups["critic"]], dim=-1)

    def sample_policy(
        self, obs: torch.Tensor | TensorDict, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample an IID action with its density in the environment action coordinates."""
        actions, info = self.actor(obs, training=training)
        return actions * self.action_support, dict(
            info, log_prob=info["log_prob"] - self.num_actions * math.log(self.action_support)
        )

    def deterministic_actions(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate policy actions with gradients and without updating normalization statistics."""
        mean, _ = self.actor.get_mean_and_std(self.actor_observations(obs), training=False)
        return mean.tanh() * self.action_support

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect using the reference truncated-zeta noise repetition process."""
        mean, std = self.actor.get_mean_and_std(self.actor_observations(obs), training=False)
        if self.noise.shape != mean.shape:
            self.noise = torch.randn_like(mean)
            self.noise_count.zero_()
        reinit = (self.noise_count == 0) | (self.noise_count >= self.noise_repeat)
        new_noise = torch.randn_like(mean)
        u = torch.rand((), device=self.device)
        new_repeat = ((u < self.noise_cdf).to(torch.int32).argmax() + 1).to(torch.int32)
        self.noise = torch.where(reinit, new_noise, self.noise)
        self.noise_repeat = torch.where(reinit, new_repeat, self.noise_repeat)
        self.noise_count = torch.where(reinit, 0, self.noise_count) + 1
        return (mean + std * self.noise).tanh() * self.action_support

    @torch.no_grad()
    def observe_rewards(self, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Update return statistics once per collected transition, including terminal rewards."""
        if not self.normalize_reward:
            return
        returns = self.gamma * self.reward_return + rewards
        self.reward_max = torch.maximum(self.reward_max, returns.abs().max())
        count = float(returns.shape[0])
        mean, variance = returns.mean(), returns.var(unbiased=False)
        if self.is_multi_gpu:
            # Center moments on the shared previous mean to avoid cancellation.
            shifted = returns - self.reward_mean
            moments = torch.stack((shifted.sum(), shifted.square().sum(), returns.new_tensor(count)))
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            count = moments[2]
            delta = moments[0] / count
            mean = self.reward_mean + delta
            variance = (moments[1] / count - delta.square()).clamp_min(0)
            dist.all_reduce(self.reward_max, op=dist.ReduceOp.MAX)
        delta = mean - self.reward_mean
        total = self.reward_count + count
        ratio = count / total
        self.reward_mean = self.reward_mean + delta * ratio
        self.reward_var = (
            self.reward_var * (self.reward_count + 1e-4) + variance * count + delta.square() * self.reward_count * ratio
        ) / total
        self.reward_count = total
        self.reward_return = returns * (~(terminated | truncated)).float()

    def auxiliary_loss(self, phase: str, batch: TensorDict) -> torch.Tensor:
        """Return the loss added to an actor or critic step; extensions override this hook."""
        return torch.zeros((), device=self.device)

    def update_auxiliary(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Update independently optimized auxiliary models after the SAC update."""
        return {}

    def _step(self, name: str, loss: torch.Tensor) -> None:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if self.use_amp and name != "temperature":
            self.grad_scaler.scale(loss).backward()
            # Reduce scaled gradients before unscaling so overflow decisions agree.
            self.reduce_parameters(optimizer)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.reduce_parameters(optimizer)
            optimizer.step()
        self.schedulers[name].step()
        if name != "temperature":
            normalize_parameters(self.models[name])

    def update(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Update actor/temperature when due, then critic and target, matching the reference order."""
        original_batch_size = batch.batch_size[0]
        if self.symmetry:
            batch = self.symmetry.augment_replay_batch(batch)
        obs, next_obs = batch["observations"], batch["next_observations"]
        actor_obs, next_actor_obs = self.actor_observations(obs), self.actor_observations(next_obs)
        critic_obs, next_critic_obs = self.critic_observations(obs), self.critic_observations(next_obs)
        reward = batch["rewards"]
        if self.normalize_reward:
            reward = reward / torch.maximum((self.reward_var + 1e-8).sqrt(), self.reward_max / self.value_max)
        metrics = {}
        if self.update_step % self.actor_update_period == 0:
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                actions_all, info = self.sample_policy(torch.cat([actor_obs, next_actor_obs], dim=0), training=True)
                actions, log_prob = actions_all.chunk(2)[0], info["log_prob"].chunk(2)[0]
                flags = [p.requires_grad for p in self.critic.parameters()]
                self.critic.requires_grad_(False)
                try:
                    qs, _ = self.critic(critic_obs, actions, training=False)
                finally:
                    for p, flag in zip(self.critic.parameters(), flags):
                        p.requires_grad_(flag)
                actor_loss = (log_prob * self.temperature().detach() - qs.min(dim=0).values).mean()
                actor_loss = actor_loss + self.auxiliary_loss("actor", batch)
                entropy = -log_prob.mean()
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_policy_loss(
                        self.deterministic_actions,
                        obs,
                        original_batch_size,
                        augmented=self.symmetry.use_data_augmentation,
                    )
                    if self.symmetry.use_mirror_loss:
                        actor_loss = actor_loss + self.symmetry.mirror_loss_coeff * symmetry_loss
                    metrics["symmetry"] = symmetry_loss
            self._step("actor", actor_loss)
            temperature = self.temperature().clone()
            temperature_loss = temperature * (entropy.detach() - self.target_entropy).mean()
            self._step("temperature", temperature_loss)
            metrics.update({
                "actor/loss": actor_loss,
                "actor/entropy": entropy,
                "actor/mean_action": actions.mean(),
                "temperature/value": temperature,
                "temperature/loss": temperature_loss,
            })
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                next_actions, info = self.sample_policy(next_actor_obs, training=False)
                entropy = self.temperature() * info["log_prob"]
                obs_all = torch.cat([critic_obs, next_critic_obs], dim=0)
                actions_all = torch.cat([batch["actions"], next_actions], dim=0)
                qs, info = self.target_critic(obs_all, actions_all, training=True)
                log_probs = _select_min_q_log_probs(qs.chunk(2, dim=1)[1], info["log_prob"].chunk(2, dim=1)[1])
                targets = _compute_categorical_td_target(
                    log_probs,
                    reward,
                    batch["terminated"].float(),
                    entropy,
                    batch["bootstrap_discount"].reshape(-1, 1),
                    self.num_bins,
                    -self.value_max,
                    self.value_max,
                )
            _, info = self.critic(obs_all, actions_all, training=True)
            critic_loss = -(targets.unsqueeze(0) * info["log_prob"].chunk(2, dim=1)[0]).sum(-1).mean()
            critic_loss = critic_loss + self.auxiliary_loss("critic", batch)
        self._step("critic", critic_loss)
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target_critic.parameters()), list(self.critic.parameters()), self.target_tau)
        # Target batch-normalization buffers are updated by its training forward, as in the pinned source.
        self.update_step += 1
        metrics.update({"critic/loss": critic_loss, "critic/max_entropy_bonus": entropy.max()})
        metrics.update(self.update_auxiliary(batch))
        if self.is_multi_gpu:
            self.synchronize_buffers()
        return {key: value.detach() for key, value in metrics.items()}

    def broadcast_parameters(self, models: Iterable[nn.Module] | None = None) -> None:
        """Initialize all replicas from rank zero, including target and temperature state."""
        if not self.is_multi_gpu:
            return
        for model in self.models.values() if models is None else models:
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)

    def reduce_parameters(self, optimizer: torch.optim.Optimizer) -> None:
        """Average this optimizer's gradients using the existing RSL-RL bucket reducer."""
        if self.is_multi_gpu:
            parameters = (p for group in optimizer.param_groups for p in group["params"])
            reduce_gradients_in_buckets(parameters, self.gpu_world_size, self.grad_reduce_bucket_mb)

    @torch.no_grad()
    def synchronize_buffers(self, models: Iterable[nn.Module] | None = None) -> None:
        """Share running buffers after local-batch training; this is not synchronized batch norm.

        Floating buffers are averaged, integer counters take the maximum. Training
        uses each rank's local minibatch statistics; inference uses shared running statistics.
        """
        if not self.is_multi_gpu:
            return
        groups = {}
        seen = set()
        for model in self.models.values() if models is None else models:
            for buffer in model.buffers():
                if id(buffer) not in seen:
                    seen.add(id(buffer))
                    groups.setdefault((buffer.device, buffer.dtype), []).append(buffer)
        for buffers in groups.values():
            flat = torch.cat([buffer.reshape(-1) for buffer in buffers])
            floating = flat.is_floating_point()
            dist.all_reduce(flat, op=dist.ReduceOp.SUM if floating else dist.ReduceOp.MAX)
            if floating:
                flat /= self.gpu_world_size
            offset = 0
            for buffer in buffers:
                count = buffer.numel()
                buffer.copy_(flat[offset : offset + count].reshape_as(buffer))
                offset += count

    def train_mode(self) -> None:
        """Enable model training; FlashSAC batch norm is controlled per forward call."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Switch online models to evaluation."""
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """Serialize learner state; replay and simulator state are intentionally separate."""
        return {
            "flash_sac_version": 1,
            "training_config": self.training_config,
            "models": {key: model.state_dict() for key, model in self.models.items()},
            "optimizers": {key: optimizer.state_dict() for key, optimizer in self.optimizers.items()},
            "schedulers": {key: scheduler.state_dict() for key, scheduler in self.schedulers.items()},
            "update_step": self.update_step,
            "action_support": self.action_support,
            "curriculum_step": self.curriculum_step,
            "grad_scaler": self.grad_scaler.state_dict(),
            "statistics": {
                key: getattr(self, key)
                for key in ("reward_return", "reward_max", "reward_mean", "reward_var", "reward_count")
            },
            "exploration": {key: getattr(self, key) for key in ("noise", "noise_count", "noise_repeat")},
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def load(self, state: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore learner state or actor weights; a runner restart uses fresh replay."""
        if state.get("flash_sac_version") != 1:
            raise ValueError("Unsupported FlashSAC checkpoint; PPO and Agile checkpoints need explicit conversion.")
        flags = (
            load_cfg if load_cfg is not None else {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        )
        unknown = set(flags) - {"actor", "critic", "optimizer", "iteration"}
        if unknown:
            raise ValueError(f"Unsupported FlashSAC load options: {sorted(unknown)}")
        if flags.get("optimizer") and not (flags.get("actor") and flags.get("critic")):
            raise ValueError("Optimizer restoration requires both actor and critic weights.")
        if flags.get("optimizer") and state["training_config"] != self.training_config:
            raise ValueError("FlashSAC training configuration differs; load weights with optimizer=False for transfer.")
        for name, model in self.models.items():
            if flags.get("actor" if name == "actor" else "critic"):
                model.load_state_dict(state["models"][name], strict=strict)
        self.action_support = state["action_support"]
        if flags.get("optimizer"):
            for name in self.optimizers:
                self.optimizers[name].load_state_dict(state["optimizers"][name])
                self.schedulers[name].load_state_dict(state["schedulers"][name])
            self.grad_scaler.load_state_dict(state["grad_scaler"])
        if flags.get("critic"):
            for key, value in state["statistics"].items():
                setattr(self, key, value.to(self.device))
        if flags.get("iteration"):
            if self.curriculum:
                self.update_curriculum(state["curriculum_step"])
            self.update_step = state["update_step"]
            for key, value in state["exploration"].items():
                setattr(self, key, value.to(self.device))
            torch.set_rng_state(state["rng"].cpu())
            if state["cuda_rng"] is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        return bool(flags.get("iteration"))

    def get_policy(self) -> nn.Module:
        """Return a deterministic TensorDict policy owning only actor parameters."""
        return FlashSACPolicy(self.actor, self.obs_groups["actor"], self.action_support)


class FlashSACPolicy(nn.Module):
    """Deterministic actor view with an explicit observation-group input contract."""

    def __init__(
        self, actor: nn.Module, obs_groups: list[str], action_support: float, concatenate_observations: bool = True
    ) -> None:
        """Initialize the deterministic actor and its input mapping."""
        super().__init__()
        self.actor = actor.as_inference() if callable(getattr(actor, "as_inference", None)) else actor
        self.obs_groups = obs_groups
        self.action_support = action_support
        self.concatenate_observations = concatenate_observations

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Compute deterministic actions without sampling or changing normalization statistics."""
        inputs = torch.cat([obs[key] for key in self.obs_groups], dim=-1) if self.concatenate_observations else obs
        mean, _ = self.actor.get_mean_and_std(inputs, training=False)
        return mean.tanh() * self.action_support

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Feed-forward policies have no recurrent state."""


def _select_min_q_log_probs(
    next_qs: torch.Tensor,  # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Select log-probs from the min-Q critic and return the next-obs half (batch, num_bins)."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]  # (B, num_bins)
    return selected


def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,)
    actor_entropy: torch.Tensor,  # (B,)
    gamma: float | torch.Tensor,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # Compute target value buckets
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    # target_bin_values
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # update indices
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)

    frac = b - lower.float()

    # Compute target probabilities using exp
    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    # Allocate output tensor
    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)

    # Scatter operations
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs


class _EntropyTemperature(nn.Module):
    """Learnable positive entropy temperature."""

    def __init__(self, initial_value: float = 0.01) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Evaluate the layer."""
        return torch.exp(self.log_temp)
