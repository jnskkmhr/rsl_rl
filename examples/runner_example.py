import torch

from rsl_rl.env.gym_env import GymEnv
from rsl_rl.runners.policy_runner import Runner
from rl_cfg import RslRlPolicyRunnerCfg, RslRlPpoCfg

from dataclasses import asdict

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TASK = "BipedalWalker-v3"


def main():
    ppo_cfg = RslRlPpoCfg(
        actor_activations=["tanh", "tanh", "linear"],
        actor_hidden_dims=[64, 64],
        actor_input_normalization=False,
        actor_noise_std=0.01,
        batch_count=1,
        clip_ratio=0.2,
        critic_activations=["tanh", "tanh", "linear"],
        critic_hidden_dims=[64, 64],
        critic_input_normalization=False,
        entropy_coeff=0.01,
        gae_lambda=0.95,
        gamma=0.99,
        gradient_clip=0.5,
        learning_rate=0.0003,
        schedule="adaptive",
        target_kl=0.01,
        value_coeff=0.5
    )
    
    cfg = RslRlPolicyRunnerCfg(
        seed=0, 
        device="cuda:0",
        agent=ppo_cfg,
        num_steps_per_env=2048,
        max_iterations=1000,
        save_interval=100,
        experiment_name="ppo",
        logger="wandb",
        wandb_project="isaaclab"
    )
    
    env = GymEnv(name=TASK, device=DEVICE, draw=True, environment_count=10)
    runner = Runner(env, asdict(cfg), log_dir="logs", device=DEVICE)
    # runner.learn(5000)


if __name__ == "__main__":
    main()
