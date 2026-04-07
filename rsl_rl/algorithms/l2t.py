# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class L2T:
    """Learn-to-Teach algorithm.

    This algorithm is a PPO-style variant that trains:
    - a teacher policy (with PPO objective) using observations from the ``critic`` observation set,
    - a value critic (for the teacher PPO objective), and
    - a student policy (with imitation + optional asymmetric PPO-style objective) using observations
      from the ``student`` observation set.
    """

    teacher: MLPModel
    """Teacher policy model."""

    critic: MLPModel
    """Value model used for teacher PPO updates."""

    student: MLPModel
    """Student policy model."""

    @staticmethod
    def _detach_hidden_state(hidden_state: torch.Tensor | tuple[torch.Tensor, ...] | None):
        """Detach hidden state tensors from autograd graph."""
        if hidden_state is None:
            return None
        if isinstance(hidden_state, tuple):
            return tuple(state.detach() for state in hidden_state)
        return hidden_state.detach()

    def __init__(
        self,
        teacher: MLPModel,
        critic: MLPModel,
        student: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        student_imitation_coef: float = 1.0,
        student_asymmetry_coef: float = 0.0,
        student_entropy_coef: float = 0.0,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        mixture_coeff: float = 0.0,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize L2T with teacher, critic, student, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Core models
        self.teacher = teacher.to(self.device)
        self.critic = critic.to(self.device)
        self.student = student.to(self.device)

        # Optimizers
        optimizer_class = resolve_optimizer(optimizer)
        self.teacher_optimizer = optimizer_class(
            chain(self.teacher.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore
        self.student_optimizer = optimizer_class(self.student.parameters(), lr=learning_rate)  # type: ignore

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO and L2T parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.student_imitation_coef = student_imitation_coef
        self.student_asymmetry_coef = student_asymmetry_coef
        self.student_entropy_coef = student_entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.mixture_coeff = mixture_coeff
        self.mixture_schedule = kwargs.get("mixture_schedule", "constant")
        self.max_iterations = kwargs.get("max_iterations", None)
        if self.mixture_schedule not in {"constant", "linear"}:
            raise ValueError(
                f"Unknown mixture schedule: {self.mixture_schedule}. Supported: 'constant', 'linear'."
            )
        self.last_student_mix_ratio = 0.0
        self.last_effective_mixture_coeff = 0.0
        self.num_updates = 0

        # Interface compatibility with runner assumptions
        self.rnd = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        self.transition.hidden_states = (
            self._detach_hidden_state(self.teacher.get_hidden_state()),
            self._detach_hidden_state(self.critic.get_hidden_state()),
        )
        self.transition.student_hidden_state = self._detach_hidden_state(self.student.get_hidden_state())

        # Run a student forward pass to populate distribution statistics and recurrent state.
        student_actions = self.student(obs, stochastic_output=True)

        teacher_actions = self.teacher(obs, stochastic_output=True)
        values = self.critic(obs)

        effective_mixture_coeff = self._get_effective_mixture_coeff()
        self.last_effective_mixture_coeff = effective_mixture_coeff

        if effective_mixture_coeff > 0.0:
            # Mix per-environment samples so only a fraction of transitions come from student actions.
            mix_mask = (torch.rand(teacher_actions.shape[0], device=self.device) < effective_mixture_coeff).unsqueeze(
                -1
            )
            actions = torch.where(mix_mask, student_actions, teacher_actions).detach()
            self.last_student_mix_ratio = mix_mask.float().mean().item()
        else:
            actions = teacher_actions.detach()
            self.last_student_mix_ratio = 0.0

        self.transition.actions = actions
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.teacher.get_output_log_prob(actions).detach()  # type: ignore[arg-type]
        self.transition.distribution_params = tuple(p.detach() for p in self.teacher.output_distribution_params)

        # Record observations before env.step()
        self.transition.observations = obs
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update normalizers."""
        self.teacher.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.student.update_normalization(obs)

        # Note: clone here because rewards can be modified by timeout bootstrapping
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore[arg-type]
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()

        self.teacher.reset(dones)
        self.critic.reset(dones)
        self.student.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        last_values = self.critic(obs).detach()

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0.0
        mean_teacher_surrogate_loss = 0.0
        mean_teacher_entropy = 0.0
        mean_student_loss = 0.0
        mean_student_imitation_loss = 0.0
        mean_student_asymmetry_loss = 0.0

        recurrent = self.teacher.is_recurrent or self.critic.is_recurrent or self.student.is_recurrent
        if recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            # Optional per-mini-batch advantage normalization
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore[arg-type]

            # Teacher forward for PPO loss terms
            if self.teacher.has_encoder and hasattr(self.teacher, "forward_encoder"):
                self.teacher.forward_encoder(
                    batch.observations,
                    encoder_state=batch.encoder_state,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
            else:
                self.teacher(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )

            actions_log_prob = self.teacher.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.teacher.output_distribution_params
            entropy = self.teacher.output_entropy

            # KL-based adaptive learning-rate schedule (teacher optimizer)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.teacher.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore[arg-type]
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for optimizer in (self.teacher_optimizer, self.student_optimizer):
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

            # Teacher PPO losses
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)  # type: ignore[arg-type]
                value_losses = (values - batch.returns).pow(2)  # type: ignore[arg-type]
                value_losses_clipped = (value_clipped - batch.returns).pow(2)  # type: ignore[arg-type]
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()  # type: ignore[arg-type]

            teacher_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Student losses
            student_actions = self.student(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.student_hidden_state,
                stochastic_output=False,
            )
            imitation_loss = nn.functional.mse_loss(student_actions, batch.actions.detach())  # type: ignore[arg-type]

            student_asymmetry_loss = torch.zeros((), device=self.device)
            if self.student_asymmetry_coef > 0.0:
                student_log_prob = self.student.get_output_log_prob(batch.actions.detach())  # type: ignore[arg-type]
                student_ratio = torch.exp(student_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
                student_asymmetry_loss = -(
                    torch.squeeze(batch.advantages)  # type: ignore[arg-type]
                    * torch.clamp(student_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                ).mean()

            student_entropy = self.student.output_entropy.mean()
            student_loss = (
                self.student_imitation_coef * imitation_loss
                + self.student_asymmetry_coef * student_asymmetry_loss
                - self.student_entropy_coef * student_entropy
            )

            # Teacher update
            self.teacher_optimizer.zero_grad()
            teacher_loss.backward()

            # Student update
            self.student_optimizer.zero_grad()
            student_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(chain(self.teacher.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.teacher_optimizer.step()

            nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.student_optimizer.step()

            # Track means
            mean_value_loss += value_loss.item()
            mean_teacher_surrogate_loss += surrogate_loss.item()
            mean_teacher_entropy += entropy.mean().item()
            mean_student_loss += student_loss.item()
            mean_student_imitation_loss += imitation_loss.item()
            mean_student_asymmetry_loss += student_asymmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_teacher_surrogate_loss /= num_updates
        mean_teacher_entropy /= num_updates
        mean_student_loss /= num_updates
        mean_student_imitation_loss /= num_updates
        mean_student_asymmetry_loss /= num_updates

        self.storage.clear()
        self.num_updates += 1

        return {
            "value": mean_value_loss,
            "surrogate": mean_teacher_surrogate_loss,
            "entropy": mean_teacher_entropy,
            "student": mean_student_loss,
            "student_imitation": mean_student_imitation_loss,
            "student_asymmetry": mean_student_asymmetry_loss,
            "student_mix_ratio": self.last_student_mix_ratio,
            "student_mixture_coeff": self.last_effective_mixture_coeff,
        }

    def _get_effective_mixture_coeff(self) -> float:
        """Return the effective rollout mixture coefficient for the current training stage."""
        if self.mixture_coeff <= 0.0:
            return 0.0

        if self.mixture_schedule == "constant":
            return self.mixture_coeff

        # Linear curriculum from 0 at the beginning to mixture_coeff at the end of training.
        if isinstance(self.max_iterations, int) and self.max_iterations > 1:
            progress = min(1.0, max(0.0, self.num_updates / float(self.max_iterations - 1)))
            return self.mixture_coeff * progress

        # Fallback to constant if total iterations are unknown.
        return self.mixture_coeff

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.teacher.train()
        self.critic.train()
        self.student.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.teacher.eval()
        self.critic.eval()
        self.student.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        return {
            "teacher_state_dict": self.teacher.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "student_state_dict": self.student.state_dict(),
            "teacher_optimizer_state_dict": self.teacher_optimizer.state_dict(),
            "student_optimizer_state_dict": self.student_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None:
            load_cfg = {
                "teacher": True,
                "critic": True,
                "student": True,
                "teacher_optimizer": True,
                "student_optimizer": True,
                "iteration": True,
            }

        if load_cfg.get("teacher"):
            teacher_state = loaded_dict.get("teacher_state_dict") or loaded_dict.get("actor_state_dict")
            if teacher_state is None:
                raise KeyError("Neither 'teacher_state_dict' nor 'actor_state_dict' was found in checkpoint.")
            self.teacher.load_state_dict(teacher_state, strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("student"):
            self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher_optimizer") and "teacher_optimizer_state_dict" in loaded_dict:
            self.teacher_optimizer.load_state_dict(loaded_dict["teacher_optimizer_state_dict"])
        if load_cfg.get("student_optimizer") and "student_optimizer_state_dict" in loaded_dict:
            self.student_optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])

        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Return the student policy for inference/export."""
        return self.student

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> L2T:
        """Construct the L2T algorithm in rsl_rl format."""
        alg_class: type[L2T] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore

        teacher_cfg_key = "teacher" if "teacher" in cfg else "actor"
        if teacher_cfg_key not in cfg:
            raise ValueError("L2T requires a 'teacher' or 'actor' model configuration.")
        student_cfg_key = "student"
        if student_cfg_key not in cfg:
            raise ValueError("L2T requires a 'student' model configuration.")

        teacher_class: type[MLPModel] = resolve_callable(cfg[teacher_cfg_key].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore

        default_sets = ["critic", "student"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # L2T is currently not compatible with RND and symmetry extensions.
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with L2T.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("The symmetry extension is not compatible with L2T.")
        cfg["algorithm"]["symmetry_cfg"] = None

        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "critic", env.num_actions, **cfg[teacher_cfg_key]).to(
            device
        )
        print(f"Teacher Model: {teacher}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = teacher.cnns  # type: ignore[attr-defined]

        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: L2T = alg_class(
            teacher,
            critic,
            student,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self.teacher.state_dict(), self.critic.state_dict(), self.student.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.teacher.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.student.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = list(chain(self.teacher.parameters(), self.critic.parameters(), self.student.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if len(grads) == 0:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
