from dataclasses import dataclass
from dataclasses import MISSING
from typing import Literal

@dataclass
class RslRlPpoCfg:
    actor_activations: list[str] = MISSING
    actor_hidden_dims: list[int] = MISSING
    critic_activations:list[str] = MISSING
    critic_hidden_dims:list[int] = MISSING
    
    # class_name: str = "PPO"
    actor_input_normalization: bool =False
    actor_noise_std: float = 0.0
    critic_input_normalization:bool =False
    
    batch_count: int = 1
    clip_ratio:float = 0.2
    entropy_coeff:float =0.0
    gae_lambda:float =0.95
    gamma:float =0.99
    gradient_clip:float =0.5
    learning_rate:float =0.0003
    schedule:str ="adaptive"
    target_kl:float =0.01
    value_coeff:float =0.5

@dataclass 
class RslRlPolicyRunnerCfg:
    seed: int = MISSING
    """The seed for the experiment. Default is 42."""

    device: str = MISSING
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""
    
    agent: RslRlPpoCfg = MISSING

    # empirical_normalization: bool = MISSING
    # """Whether to use empirical normalization."""

    # policy: RslRlPpoActorCriticCfg = MISSING
    # """The policy configuration."""

    # algorithm: RslRlPpoAlgorithmCfg = MISSING
    # """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
    
    class_name: str = "PPO"
    
if __name__ == "__main__":
    from dataclasses import asdict
    
    ppo_cfg = RslRlPpoCfg(
        actor_activations=["tanh", "tanh", "linear"],
        actor_hidden_dims=[64, 64],
        actor_input_normalization=False,
        actor_noise_std=0.0,
        batch_count=1,
        clip_ratio=0.2,
        critic_activations=["tanh", "tanh", "linear"],
        critic_hidden_dims=[64, 64],
        critic_input_normalization=False,
        entropy_coeff=0.0,
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
    print(asdict(cfg))