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
        squash_output=False,
        initializer:str = "glorot_uniform",
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
        
        if initializer == "zeros":
            self.init_sequential_weights_zero(self.actor)
        elif initializer == "xavier_uniform":
            self.init_sequential_weights_xavier_uniform(self.actor)
        elif initializer == "xavier_normal":
            self.init_sequential_weights_xavier_normal(self.actor)
        elif initializer == "kaiming_uniform":
            self.init_sequential_weights_kaiming_uniform(self.actor)
        elif initializer == "kaiming_normal":
            self.init_sequential_weights_kaiming_normal(self.actor)
        elif initializer == "orthogonal":
            self.init_sequential_weights_orthogonal(self.actor)

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
        
        # small number to avoid numerical issues
        self.eps = 1e-6
        self.squash_output = squash_output
    
    
    # weight initializers
    # Initialize weights of last layer to zero same as https://arxiv.org/abs/1812.06298
    def init_sequential_weights_zero(self, sequential):
        layer = sequential[-1]
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
                
    def init_sequential_weights_xavier_uniform(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def init_sequential_weights_xavier_normal(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def init_sequential_weights_kaiming_uniform(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def init_sequential_weights_kaiming_normal(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def init_sequential_weights_orthogonal(self, sequential):
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def init_weights_zero(self, network):
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
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # # squash mean begore putting it to gaussian
        # mean = torch.tanh(mean) 
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            std = torch.clamp(std, STD_MIN, STD_MAX)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
            std = torch.clamp(std, STD_MIN, STD_MAX)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        if self.squash_output:
            gaussian_action = self.distribution.sample()
            return torch.tanh(gaussian_action)
        else:
            return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        # actions_mean = torch.tanh(actions_mean) # squash mean
        if self.squash_output:
            return torch.tanh(actions_mean)
        else:
            return actions_mean
    
    def get_actions_log_prob(self, actions):
        if self.squash_output:
            # tanh^-1(action) = gaussian_action
            gaussian_action = TanhBijector.inverse(actions)
            log_prob = self.distribution.log_prob(gaussian_action).sum(dim=-1)
            # change of variable formula from SAC paper: https://arxiv.org/abs/1801.01290
            
            # SB3 implementation
            log_prob = log_prob - torch.sum(torch.log(1 - actions**2 + self.eps), dim=-1)
            
            # OpenAI spinningup implementation
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            # log_prob -= (2*(math.log(2) - gaussian_action - F.softplus(-2*gaussian_action))).sum(dim=1)
            return log_prob
        else:
            return self.distribution.log_prob(actions).sum(dim=-1)

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

class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)
