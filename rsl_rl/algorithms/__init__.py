from .agent import Agent
from .d4pg import D4PG
from .ddpg import DDPG
from .dppo import DPPO
from .dsac import DSAC
from .ppo import PPO
from .sac import SAC
from .td3 import TD3

__all__ = ["Agent", "DDPG", "D4PG", "DPPO", "DSAC", "PPO", "SAC", "TD3"]

ALG_FACTORY = {
    "DDPG": DDPG,
    "D4PG": D4PG,
    "DPPO": DPPO,
    "DSAC": DSAC,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}