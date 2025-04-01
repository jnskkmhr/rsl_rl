# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

STD_MIN = math.exp(-20)
STD_MAX = math.exp(0.1)
LOG_STD_MAX = 0.1
LOG_STD_MIN = -20

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        initializer:str = "xavier_uniform",
        init_last_layer_zero: bool = False,
        clip_actions=False,
        clip_actions_range: tuple = (-1.0, 1.0),
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)

        self.actor = nn.Sequential(*actor_layers)
        
        self.eps = 1e-6
        self.clip_actions = clip_actions
        self.clip_actions_range = clip_actions_range
        if self.clip_actions:
            self.clipping_layer = nn.Tanh()
        
        if initializer == "xavier_uniform":
            self.init_sequential_weights_xavier_uniform(self.actor)
        elif initializer == "xavier_normal":
            self.init_sequential_weights_xavier_normal(self.actor)
        elif initializer == "kaiming_uniform":
            self.init_sequential_weights_kaiming_uniform(self.actor)
        elif initializer == "kaiming_normal":
            self.init_sequential_weights_kaiming_normal(self.actor)
        elif initializer == "orthogonal":
            self.init_sequential_weights_orthogonal(self.actor)
        
        # with residual learning setup
        if init_last_layer_zero:
            self.init_layer_zero(self.actor[-1])

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
    
    
    # weight initializers
    def init_sequential_weights_xavier_uniform(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.xavier_uniform_(layer.bias.view(1, -1))
    
    def init_sequential_weights_xavier_normal(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.xavier_normal_(layer.bias.view(1, -1))
    
    def init_sequential_weights_kaiming_uniform(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.kaiming_uniform_(layer.bias.view(1, -1))
    
    def init_sequential_weights_kaiming_normal(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.kaiming_normal_(layer.bias.view(1, -1))
    
    def init_sequential_weights_orthogonal(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.orthogonal_(layer.bias.view(1, -1))
    
    # Initialize weights of last layer to zero same as https://arxiv.org/abs/1812.06298
    def init_layer_zero(self, network):
        if isinstance(network, nn.Linear):
            nn.init.zeros_(network.weight)
            if network.bias is not None:
                nn.init.zeros_(network.bias)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        mode = self.distribution.mean
        if self.clip_actions:
            mode = ((mode + 1) /2.0)* (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return mode

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def actions_distribution(self) -> torch.Tensor:
        # Mean and Std concatenated on an extra dimension
        return torch.stack([self.distribution.mean, self.distribution.stddev], dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def build_distribution(self, parameters):
        # build the distribution
        return Normal(parameters[..., 0], parameters[..., 1])

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        if self.clip_actions:
            mean = self.clipping_layer(mean)

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        act = self.distribution.sample()
        if self.clip_actions:
            # Apply tanh to clip the actions to [-1, 1]
            act = self.clipping_layer(act)
            # Rescale the actions to the desired range
            act = ((act + 1) / 2.0) * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return act
    
    def get_actions_log_prob(self, actions):
        # Scale the actions to [-1, 1] before computing the log probability.
        if self.clip_actions:
            # The unscaled actions still have the tanh applied to them.
            unscaled_actions = (actions - self.clip_actions_range[0]) / (self.clip_actions_range[1] - self.clip_actions_range[0]) * 2.0 - 1.0
            # Revert the tanh to get the original actions. We use the TanhBijector to avoid numerical issues.
            gaussian_actions = self.inverse_tanh(unscaled_actions)
            return (self.distribution.log_prob(gaussian_actions) - torch.log(1 - unscaled_actions*unscaled_actions + 1e-6)).sum(dim=-1)
        else:
            return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        mode= self.actor(observations)
        if self.clip_actions:
            mode = ((mode + 1) / 2.0) * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return mode

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
    
    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())
    
    @staticmethod
    def inverse_tanh(y):
        eps = torch.finfo(y.dtype).eps
        return ActorCritic.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))
