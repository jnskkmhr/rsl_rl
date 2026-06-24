# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_ae import PPOAE
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


class PPOVAE(PPOAE):
    r"""PPO with a beta-VAE structured latent regularization.

    Extends :class:`PPOAE` by adding the KL divergence between the
    encoder posterior :math:`q(z \mid x)` and the isotropic Gaussian prior
    :math:`\mathcal{N}(0, I)` to the training objective:

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{PPO}}
                    + \lambda_{\text{dec}} \, \mathcal{L}_{\text{recon}}
                    + \lambda_{\text{KL}} \, \mathcal{L}_{\text{KL}}

    where

    .. math::

        \mathcal{L}_{\text{KL}}
            = \frac{1}{d_z} \sum_{i=1}^{d_z}
              \max\!\Bigl(
                  \tau,\;
                  -\tfrac{1}{2}(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2)
              \Bigr)

    The optional *free-nats* threshold :math:`\tau` (``kl_clip``) prevents
    posterior collapse for small latent dimensions by only penalising KL above
    the tolerance.

    Note:
        This class is designed for use with :class:`~rsl_rl.models.MLPVAEModel`
        (or any actor with ``has_decoder = True``, a ``get_decoder_output()``
        method, and a ``get_vae_params()`` method returning ``(mu, log_var)``).
        If the actor has no decoder, an error is raised. If it has a decoder but
        no ``get_vae_params()``, the class falls back to plain
        :class:`PPOAE` behaviour (no KL term).
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
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
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Decoder parameters
        decoder_loss_coef: float = 1.0,
        loss_type: str = "mse",
        # VAE parameters
        kl_loss_coef: float = 1.0,
        kl_clip: float = 0.0,
    ) -> None:
        """Initialize PPOVAE.

        Args:
            actor: Actor model (should be :class:`~rsl_rl.models.MLPVAEModel`).
            critic: Critic model.
            storage: Rollout storage.
            num_learning_epochs: Number of PPO learning epochs per update.
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
            decoder_loss_coef: Coefficient for the decoder reconstruction loss.
            loss_type: Regression loss type for decoder. Supported: ``"mse"``, ``"huber"``.
            kl_loss_coef: Coefficient for the VAE KL loss.
            kl_clip: Free-nats tolerance. Only KL per dimension above
                this threshold is penalised. Set to ``0.0`` to disable (standard VAE).
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
            decoder_loss_coef=decoder_loss_coef,
            loss_type=loss_type,
        )

        self.kl_loss_coef = kl_loss_coef
        self.kl_clip = kl_clip

    def update(self) -> dict[str, float]:
        """Run PPO update epochs and add decoder reconstruction and KL losses.

        Extends :meth:`PPOEncoderDecoder.update` by computing the per-dimension KL
        divergence from the VAE encoder and adding it (optionally with a free-nats
        threshold) to the combined loss.

        Returns:
            Dict with all PPO loss keys plus ``"decoder"`` and ``"kl"``.
        """
        if not getattr(self.actor, "has_decoder", False):
            raise RuntimeError("Actor has no decoder")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_decoder_loss = 0.0
        mean_kl_loss = 0.0
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

            # Symmetric augmentation
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

            # Forward pass — also re-runs encoder and decoder for aux losses
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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

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

            # Decoder reconstruction loss
            decoder_output = self.actor.get_decoder_output()
            if decoder_output is not None and self.decoder_loss_coef > 0:
                with torch.no_grad():
                    decoder_target = torch.cat(
                        [batch.observations[g] for g in self.actor.decoder_obs_groups],
                        dim=-1,
                    )
                decoder_loss = self.loss_fn(decoder_output, decoder_target)
                loss = loss + self.decoder_loss_coef * decoder_loss
            else:
                decoder_loss = torch.zeros((), device=self.device)

            # VAE KL divergence loss: KL(q(z|x) || N(0, I))
            vae_params = self.actor.get_vae_params() if hasattr(self.actor, "get_vae_params") else None
            if vae_params is not None and self.kl_loss_coef > 0:
                mu, log_var = vae_params
                # Per-dimension KL: -0.5 * (1 + log_var - mu^2 - exp(log_var))
                kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
                if self.kl_clip > 0.0:
                    kl_per_dim = torch.clamp(kl_per_dim, min=self.kl_clip)
                kl_loss = kl_per_dim.mean()
                loss = loss + self.kl_loss_coef * kl_loss
            else:
                kl_loss = torch.zeros((), device=self.device)

            # Backprop
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
            mean_decoder_loss += decoder_loss.item()
            mean_kl_loss += kl_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_decoder_loss /= num_updates
        mean_kl_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "decoder": mean_decoder_loss,
            "kl": mean_kl_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        return loss_dict
