# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPODistillation(PPO):
    r"""PPO with an imitation loss from a frozen teacher policy.

    Trains the student (actor) jointly with:

    - **PPO surrogate loss** — on-policy RL using returns and advantages computed from the
      student's own rollouts and value function.
    - **Imitation loss** — student mean actions match the teacher's privileged actions stored
      in the rollout buffer (behavior cloning term).
    - **Encoder reconstruction loss** (optional) — student encoder output matches the teacher
      encoder output stored in the rollout buffer.
    - **Decoder reconstruction loss** (optional) — student decoder output matches the decoder
      target observations (active only when the actor has ``has_decoder = True``).

    The total loss per mini-batch is:

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{PPO}} +
                      \lambda_{\text{imit}} \mathcal{L}_{\text{imit}} +
                      \lambda_{\text{enc}} \mathcal{L}_{\text{enc}} +
                      \lambda_{\text{dec}} \mathcal{L}_{\text{dec}}

    where :math:`\mathcal{L}_{\text{PPO}} = \mathcal{L}_{\text{surrogate}} +
    c_v \mathcal{L}_{\text{value}} - c_e \mathcal{H}`.

    The teacher is always kept in eval mode with all parameters frozen.
    """

    teacher: MLPModel
    """The frozen teacher (privileged) model."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded from a checkpoint."""

    def __init__(
        self,
        actor: MLPModel,
        teacher: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # PPO parameters
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # Distillation parameters
        imitation_loss_coef: float = 1.0,
        encoder_loss_coef: float = 0.0,
        decoder_loss_coef: float = 0.0,
        loss_type: str = "mse",
        loss_schedule: str = "fixed",
        total_iteration: int = 10_000,
    ) -> None:
        """Initialize PPODistillation.

        Args:
            actor: Student model that interacts with the environment.
            teacher: Frozen privileged model used as imitation target. Parameters are
                immediately frozen (``requires_grad = False``).
            critic: Critic model for value function estimation.
            storage: Rollout storage (must be of type ``"ppo_distillation"``).
            num_learning_epochs: Number of PPO update epochs per rollout.
            num_mini_batches: Number of mini-batches per epoch.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda.
            value_loss_coef: Weight of the value function loss.
            entropy_coef: Weight of the entropy bonus.
            learning_rate: Initial learning rate.
            max_grad_norm: Maximum gradient norm for clipping.
            optimizer: Optimizer name (``"adam"`` or ``"sgd"``).
            use_clipped_value_loss: Whether to use the clipped value loss.
            schedule: Learning rate schedule (``"adaptive"`` or ``"fixed"``).
            desired_kl: Target KL divergence for adaptive LR scheduling.
            normalize_advantage_per_mini_batch: Normalize advantages within each mini-batch.
            device: Torch device string.
            rnd_cfg: Optional Random Network Distillation configuration dict.
            symmetry_cfg: Optional symmetry augmentation configuration dict.
            multi_gpu_cfg: Optional multi-GPU configuration dict.
            imitation_loss_coef: Weight of the behavior cloning loss.
            encoder_loss_coef: Weight of the encoder reconstruction loss. Set to ``0`` to
                disable (default).
            decoder_loss_coef: Weight of the decoder reconstruction loss. Set to ``0`` to
                disable (default). Only active when the actor has ``has_decoder = True``.
            loss_type: Regression loss for imitation / encoder / decoder terms. Supported:
                ``"mse"``, ``"huber"``.
        """
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.teacher = teacher.to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.imitation_loss_coef = imitation_loss_coef
        self.encoder_loss_coef = encoder_loss_coef
        self.decoder_loss_coef = decoder_loss_coef

        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type not in loss_fn_dict:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")
        self.loss_fn = loss_fn_dict[loss_type]

        self.total_iteration = total_iteration
        self.loss_schedule = loss_schedule
        self.current_iteration = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and record teacher privileged actions."""
        actions = super().act(obs)
        # Teacher inference (no grad, deterministic)
        self.transition.privileged_actions = self.teacher(obs).detach()
        teacher_enc = self.teacher.get_encoder_state()
        self.transition.privileged_encoder_state = teacher_enc.detach() if teacher_enc is not None else None
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record environment step and reset teacher recurrent state on episode ends."""
        super().process_env_step(obs, rewards, dones, extras)
        self.teacher.reset(dones)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run PPO + imitation update epochs and return mean losses."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_imitation_loss = 0.0
        mean_encoder_loss = 0.0
        mean_decoder_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Symmetry augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Student forward pass
            if self.actor.has_encoder and hasattr(self.actor, "forward_encoder"):
                self.actor.forward_encoder(
                    batch.observations,
                    encoder_state=batch.encoder_state,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
            else:
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
            # curriculum from https://arxiv.org/pdf/2602.15827
            if self.loss_schedule == "curriculum":
                weight_distillation = max(0.1, 1 - self.current_iteration / (self.total_iteration / 2))
                weight_ppo = 1 - weight_distillation
            elif self.loss_schedule == "fixed":
                weight_distillation = 1.0
                weight_ppo = 1.0
            else:
                weight_distillation = 1.0
                weight_ppo = 1.0

            loss = torch.zeros((), device=self.device)

            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
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
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()  # type: ignore

            loss += weight_ppo * (
                surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            )

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss_fn = torch.nn.MSELoss()
                symmetry_loss = mse_loss_fn(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mse_loss_fn = torch.nn.MSELoss()
                rnd_loss = mse_loss_fn(predicted_embedding, target_embedding)

            # action imitation loss
            imitation_loss = torch.zeros((), device=self.device)
            if self.imitation_loss_coef > 0 and batch.privileged_actions is not None:
                # distribution_params[0] is the mean for a Gaussian distribution
                student_actions_mean = distribution_params[0]
                imitation_loss = self.loss_fn(student_actions_mean, batch.privileged_actions)
                loss += weight_distillation * self.imitation_loss_coef * imitation_loss

            # encoder and decoder imitation loss
            encoder_loss = torch.zeros((), device=self.device)
            decoder_loss = torch.zeros((), device=self.device)
            student_encoder_state = None
            if batch.encoder_state is not None and batch.privileged_encoder_state is not None:
                # Encoder matching loss
                if self.encoder_loss_coef > 0:
                    student_encoder_state = self.actor.get_encoder_state()
                    encoder_loss = self.loss_fn(student_encoder_state, batch.privileged_encoder_state)
                    loss += self.encoder_loss_coef * encoder_loss

                # Decoder matching loss
                if self.decoder_loss_coef > 0:
                    with torch.no_grad():
                        teacher_decoder_output = self.teacher.get_decoder_inference(batch.privileged_encoder_state)
                    student_decoder_output = self.teacher.get_decoder_inference(student_encoder_state)
                    decoder_loss = self.loss_fn(student_decoder_output, teacher_decoder_output)
                    loss += self.decoder_loss_coef * decoder_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_imitation_loss += imitation_loss.item()
            mean_encoder_loss += encoder_loss.item()
            mean_decoder_loss += decoder_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_imitation_loss /= num_updates
        mean_encoder_loss /= num_updates
        mean_decoder_loss /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        loss_dict = {
            "imitation": mean_imitation_loss,
            "encoder_reconstruction": mean_encoder_loss,
            "decoder_reconstruction": mean_decoder_loss,
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "weight_ppo": weight_ppo,
            "weight_distillation": weight_distillation,
        }
        if self.rnd and mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if self.symmetry and mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        self.current_iteration += 1

        return loss_dict

    def train_mode(self) -> None:
        """Set student and critic to train mode; keep teacher in eval mode."""
        super().train_mode()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set all models to eval mode."""
        super().eval_mode()
        self.teacher.eval()

    # def save(self) -> dict:
    #     """Return a dict of all models for saving."""
    #     saved = super().save()
    #     saved["teacher_state_dict"] = self.teacher.state_dict()
    #     return saved

    # def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    #     """Load specified models from a saved dict."""
    #     if load_cfg is None and any("actor_state_dict" in k for k in loaded_dict):
    #         # Loading from a PPO / privileged-policy checkpoint: only populate teacher
    #         load_cfg = {"teacher": True, "iteration": False}  # during training
    #     elif load_cfg is None:
    #         load_cfg = {
    #             "actor": True,
    #             "critic": True,
    #             "teacher": True,
    #             "optimizer": True,
    #             "iteration": True,
    #         }
    #     load_cfg = {"actor": True, "teacher": True, "iteration": False}  # during inference
    #     load_iteration = super().load(loaded_dict, load_cfg, strict)

    #     if load_cfg.get("teacher"):
    #         self.teacher.load_state_dict(
    #             loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
    #         )
    #         self.teacher_loaded = True

    #     return load_iteration

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "student_actor_state_dict": self.actor.state_dict(),
            "student_critic_state_dict": self.critic.state_dict(),
            "student_optimizer_state_dict": self.optimizer.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None and "actor_state_dict" in loaded_dict:
            # Loading from a PPO / privileged-policy checkpoint: populate teacher and
            # warm-start student actor MLP + critic from teacher weights.
            load_cfg = {"teacher": True, "init_student_from_teacher": True, "iteration": False}
        elif load_cfg is None:
            # Loading from a PPO distillation checkpoint (resume training or inference).
            load_cfg = {
                "actor": True,
                "critic": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["student_actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["student_critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        teacher_dict = loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"]
        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(teacher_dict, strict=strict)
            self.teacher_loaded = True
        if load_cfg.get("init_student_from_teacher"):
            # Copy shared MLP weights (mlp, distribution) from teacher to student actor.
            # The encoder is excluded — different architecture (TCN vs VAE), learned from scratch.
            student_dict = self.actor.state_dict()
            for key, value in teacher_dict.items():
                if key.startswith("mlp.") or key.startswith("distribution."):
                    if key in student_dict and student_dict[key].shape == value.shape:
                        student_dict[key] = value
            self.actor.load_state_dict(student_dict, strict=False)
            # Warm-start critic from teacher checkpoint if available
            critic_src = loaded_dict.get("critic_state_dict")
            if critic_src is not None:
                self.critic.load_state_dict(critic_src, strict=strict)

        return load_cfg.get("iteration", False)

    def get_teacher(self) -> MLPModel:
        """Return the teacher model."""
        return self.teacher

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (teacher included for consistent init)."""
        super().broadcast_parameters()
        model_params = [self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.teacher.load_state_dict(model_params[0])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPODistillation:
        """Construct the PPODistillation algorithm from a config dict.

        Expected top-level keys in ``cfg``:
        - ``"algorithm"``: algorithm hyperparameters (including ``class_name``).
        - ``"student"``: student model config (including ``class_name``).
        - ``"teacher"``: teacher model config (including ``class_name``).
        - ``"critic"``: critic model config (including ``class_name``).
        - ``"obs_groups"``: observation group mapping.
        - ``"num_steps_per_env"``: rollout length per environment.
        - ``"multi_gpu"``: multi-GPU config or ``None``.
        """
        alg_class: type[PPODistillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["student", "teacher", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")

        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
            device
        )
        print(f"Teacher Model: {teacher}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = student.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "ppo_distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg: PPODistillation = alg_class(
            student, teacher, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        return alg
