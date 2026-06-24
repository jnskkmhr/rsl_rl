# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPOAE(PPO):
    r"""PPO with an auxiliary auto-encoder reconstruction loss.

    Extends :class:`PPO` by adding a supervised reconstruction objective for actors that have a
    decoder (``actor.has_decoder is True``). After the standard PPO update the decoder is trained
    to reconstruct the encoder observations from the encoder bottleneck latent.

    The decoder loss is computed as:

    .. math::

        \\mathcal{L}_{\\text{dec}} = \\text{loss\\_fn}(\\hat{o}_{\\text{enc}}, o_{\\text{enc}})

    where :math:`\\hat{o}_{\\text{enc}}` is the decoder output and :math:`o_{\\text{enc}}` is the
    original (normalised or raw) encoder observation used as the reconstruction target.

    The decoder loss is **added to the PPO loss** so that a single ``optimizer.step()`` updates
    both the encoder/decoder and the policy MLP jointly. A separate coefficient
    ``decoder_loss_coef`` scales its contribution.

    Note:
        This class is only meaningful when the actor is an :class:`~rsl_rl.models.MLPAutoEncoderModel`
        (or any model with ``has_decoder = True`` and a ``get_decoder_output()`` method).
        If the actor has no decoder, the class behaves identically to :class:`PPO`.
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
    ) -> None:
        """Initialize PPOAE.

        Args:
            actor: Actor model (should be an :class:`~rsl_rl.models.MLPAutoEncoderModel`).
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
            loss_type: Type of regression loss for the decoder. Supported: ``"mse"``, ``"huber"``.
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

        self.decoder_loss_coef = decoder_loss_coef

        # Resolve loss function (same pattern as Distillation)
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type not in loss_fn_dict:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")
        self.loss_fn = loss_fn_dict[loss_type]

    def update(self) -> dict[str, float]:
        """Run PPO update epochs and add decoder reconstruction loss.

        The decoder loss is computed as the regression error between the actor's decoder output
        and the original (concatenated) encoder observations used as reconstruction targets.
        It is accumulated alongside the standard PPO losses and returned in the loss dict.

        Returns:
            Dict with all PPO loss keys plus ``"decoder"`` if the actor has a decoder.
        """
        if not getattr(self.actor, "has_decoder", False):
            raise RuntimeError("Actor has no decoder")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        # Decoder loss
        mean_decoder_loss = 0.0
        # RND loss
        mean_rnd_loss = 0.0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
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
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
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
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss_fn = torch.nn.MSELoss()
                symmetry_loss = mse_loss_fn(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )

                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mse_loss_fn = torch.nn.MSELoss()
                rnd_loss = mse_loss_fn(predicted_embedding, target_embedding)

            # Decoder reconstruction loss
            decoder_output = self.actor.get_decoder_output()
            if decoder_output is not None and self.decoder_loss_coef > 0:
                # Build the reconstruction target: normalized encoder observations.
                # The encoder receives normalized obs, so the decoder must reconstruct the
                # same normalized space — not the raw obs (which would force the decoder to
                # also learn the inverse normalizer).
                with torch.no_grad():
                    decoder_target = torch.cat(
                        [batch.observations[g] for g in self.actor.decoder_obs_groups],
                        dim=-1,
                    )
                    # decoder_target = self.actor.encoder_obs_normalizer(decoder_target)
                decoder_loss = self.loss_fn(decoder_output, decoder_target)
                loss = loss + self.decoder_loss_coef * decoder_loss
            else:
                decoder_loss = torch.zeros((), device=self.device)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # Decoder loss
            mean_decoder_loss += decoder_loss.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_decoder_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "decoder": mean_decoder_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        return loss_dict
