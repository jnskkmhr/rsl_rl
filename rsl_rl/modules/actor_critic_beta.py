# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta

from rsl_rl.utils import resolve_nn_activation


class ActorCriticBeta(nn.Module):
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
                self.alpha = nn.Linear(actor_hidden_dims[layer_index], num_actions)
                self.beta = nn.Linear(actor_hidden_dims[layer_index], num_actions)
                self.alpha_activation = nn.Softplus()
                self.beta_activation = nn.Softplus()
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
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
            self.init_layer_zero(self.alpha)
            self.init_layer_zero(self.beta)

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

        # Action distribution (populated in update_distribution)
        self.distribution = None
        self.a = None
        self.b = None
        # disable args validation for speedup
        Beta.set_default_validate_args(False)
        
        self.eps = 1e-6

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
    
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
        action_mean = self.a / (self.a + self.b + 1e-6) # type: ignore
        return self.scale_action(action_mean)
    
    @property
    def action_std(self):
        return torch.sqrt(self.a * self.b / ((self.a + self.b + 1) * (self.a + self.b) ** 2)) # type: ignore

    @property
    def actions_distribution(self):
        # Alpha and beta concatenated on an extra dimension
        return torch.stack([self.a, self.b], dim=-1) # type: ignore
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def build_distribution(self, parameters):
        # create distribution
        return Beta(parameters[...,0], parameters[...,1])

    def update_distribution(self, observations):
        # compute mean
        latent = self.actor(observations)
        self.a = self.alpha_activation(self.alpha(latent)) + 1.0 + self.eps
        self.b = self.beta_activation(self.beta(latent)) + 1.0 + self.eps
        # create distribution
        self.distribution = Beta(self.a, self.b)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        beta_action = self.distribution.sample()
        return self.scale_action(beta_action)

    def get_actions_log_prob(self, actions):
        beta_actions = self.unscale_action(actions)
        return self.distribution.log_prob(beta_actions).sum(dim=-1)
        # return self.distribution.log_prob(beta_actions).sum(dim=-1) - torch.log(torch.tensor(2.0, device=actions.device))

    def act_inference(self, observations):
        self.update_distribution(observations)
        return self.action_mean

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
    
    def scale_action(self, action):
        return 2*action - 1
    
    def unscale_action(self, action):
        return (action + 1) / 2