# RSL-RL

The `sac-pr-enhancement-compatible` branch merges `review/sac-pr-enhancement` into
`jnskkmhr/supermimic` as fork build `5.5.2+supermimic.1`. It retains PPO,
distillation, SuperMimic, FlashSAC and FastSAC, and adds the new native `SAC`.

`OffPolicyRunner` selects the existing FlashSAC/FastSAC collection loop for those
algorithms and the standard runner loop for native SAC. Existing model configs
retain `extra_input_dim`, `layer_norm`, ensemble models and the legacy squashed
Gaussian parameters. Native SAC uses separate Q models and its own replay buffer;
changing an existing task to SAC requires an SAC algorithm/model configuration.
Existing PPO and SuperMimic tasks do not change algorithms when upgrading.

See [FlashSAC usage and compatibility](docs/flash_sac.md) for the legacy off-policy path.

**RSL-RL** is a GPU-accelerated, lightweight learning library for robotics research. Its compact design allows
researchers to prototype and test new ideas without the overhead of modifying large, complex libraries. RSL-RL can also
be used out-of-the-box by installing it via [PyPI](https://pypi.org/project/rsl-rl-lib/), supports multi-GPU training,
and features common algorithms for robot learning.

> Additional algorithms, models, and more, mainly contributed by the community, can be found on the
> [`extras`](https://github.com/leggedrobotics/rsl_rl/tree/extras) branch. This allows us to keep the core library on
> `main` minimal and easy to start with. To see a list of the additional features, please check the
> [Extras Documentation](https://leggedrobotics.github.io/rsl_rl/extras/).

## Key Features

- **Minimal, readable codebase** with clear extension points for rapid prototyping.
- **Robotics-first methods** including PPO, SAC, and Student-Teacher Distillation.
- **High-throughput training** with native Multi-GPU support.
- **Proven performance** in numerous research publications.

## Learning Environments

RSL-RL is currently used by the following robot learning libraries:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (built on top of NVIDIA Isaac Sim)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym) (built on top of NVIDIA Isaac Gym)
- [mjlab](https://github.com/mujocolab/mjlab) (built on top of MuJoCo Warp)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (built on top of MuJoCo MJX and Warp)

## Installation

Before installing RSL-RL, ensure that Python `3.9+` is available. It is recommended to install the library in a virtual
environment (e.g. using `venv` or `conda`), which is often already created by the used environment library (e.g.
Isaac Lab). If so, make sure to activate it before installing RSL-RL.

### Installing RSL-RL as a dependency

```bash
pip install rsl-rl-lib
```

### Installing RSL-RL for development

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

## Citation

If you use RSL-RL in your research, please cite the [paper](https://arxiv.org/abs/2509.10771):

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```

## Holosoma FastSAC branch

`FastSAC` uses Holosoma's independent distributional critic targets, mean-Q actor
objective, AdamW optimizers, IID exploration and empirical observation normalization.
`FlashSAC` remains available for comparisons. Both use the native off-policy runner
and its replay implementation. Timeouts bootstrap from final observations; true
terminations do not. Replay returns stop at the first episode boundary and never
wrap into stale rewards. Observation normalization uses the correct pooled variance.

The `jnskkmhr/holosoma` branches of this library and `compositional_rsl_rl` must be
used together. The compositional library provides `CompositionalFastSACAlgorithmCfg`
and `CompositionalFastSACRunnerCfg`, plus a head configuration example in its README.
No IsaacLab dependency pins or task registries are changed by these branches.

### SuperMimic (experimental branch)

This branch ports [SuperMimic PR #255](https://github.com/leggedrobotics/rsl_rl/pull/255)
from revision `b7ed5e55aec0f45b89e9410dcdf9fa7df01516ad`.
`SuperMimic` uses the standard runner interaction loop, sequence replay, a learned
physical world model, differentiable task rewards, and imagined actor/value updates.
The environment must supply noise-free state/goal/context/label observation sets and
`extras["final_obs_every_step"] = True` with physical successors captured before resets
and external events. Invalid true-terminal successors are excluded from dynamics
training; invalid continuing or timeout states fail explicitly.

Torchrun supports one process per GPU across nodes. Actor, dynamics and critic
parameters, gradients and observation moments synchronize; replay and actor histories
remain local. World-model and policy batch sizes are global by default and must be
divisible by world size. Replay capacity is per process. Checkpoints contain all three
models and optimizers; replay is re-warmed after resume.
