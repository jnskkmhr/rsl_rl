# Native FlashSAC

This fork supplies `rsl_rl.algorithms.FlashSAC` and `rsl_rl.runners.OffPolicyRunner`.
The core algorithm and runner have no IsaacLab, `compositional_rsl_rl`, or `flash_rl` dependency.
They require explicitly supplied policy and critic models. The reference network architecture is
provided by `compositional_rsl_rl`, not by the core package.
Installing the compositional FlashSAC branch automatically installs its pinned revision of this fork.
For core development, install this checkout with `python -m pip install -e /path/to/rsl_rl`.
The local version `5.5.1+flashsac.3` identifies this development fork, not an upstream release.

## Source and numerical contract

The temperature, categorical projection, optimizer order,
target updates, and repeated exploration noise are ported from the NVIDIA FlashSAC fork
at `11a276cad45ba8816639f66c210dfc2db755bb2c`, the revision pinned by Agile.
Agile records original upstream revision `87edc9061150ae9e962dd84e6544e27a1554b3ab`.
MIT and Apache replay notices are retained in `licenses/dependencies/FlashSAC*.txt`
and included in the distribution.

The compositional package's `tests/reference/flash_sac_parity.py` compares eight updates with the pinned source,
including all model parameters/buffers and loss metrics. Put that checkout, the compositional package and this fork
on `PYTHONPATH` and run it with `--device cpu`, `--device cuda:0`, or
`--device cuda:0 --amp`. The reference package is needed only for this comparison.
The regular test suite has no reference-package dependency.

Relative to the pinned source, replay is an owned TensorDict ring with arbitrary n-step
horizons and a memory budget. Short sequences use their actual `gamma**k`.
Collection-time reward statistics include the ending reward before clearing the return,
matching Agile's correction. Replay sampling does not update those statistics.
Native checkpoints and RSL-RL logging replace the reference runner infrastructure.

## Configuration

`FlashSAC(actor, critic, num_actions, obs_groups, ...)` accepts supplied models directly.
`construct_algorithm` resolves explicit `actor.class_name` and `critic.class_name` settings
using the existing RSL-RL class resolver. Neither architecture has an implicit default.
Constructors receive `(obs, obs_groups, obs_set, num_actions, **model_cfg)`.
The existing `MLPModel` accepts optional `extra_input_dim`/`extra_input` arguments for a runtime
input such as a Q action, and `MLP` accepts `hidden_dims=[]` for a linear projection. Squashed
Gaussian and categorical value distributions are selectable through the normal distribution
interface. Residual and normalized ensemble architecture remains in compositional RSL-RL.

The model interface is:

| Capability | Contract |
| --- | --- |
| Actor parameters | `get_mean_and_std(observations, training)` returns pre-tanh Gaussian mean and standard deviation. |
| Actor sampling | `forward(observations, training)` returns reparameterized tanh actions and `{"log_prob": ...}` with shape `[batch]`, including the tanh Jacobian. |
| Critic support | `value_support` exposes uniformly spaced bins from `-value_max` to `value_max`, shape `[num_bins]`. |
| Critic evaluation | `forward(observations, actions, training)` returns Q values `[ensemble, batch]` and `{"log_prob": ...}` with shape `[ensemble, batch, num_bins]`. |
| Optional parameter constraints | Modules can implement `normalize_parameters()`; the learner invokes this after initialization and optimizer steps. |
| Optional inference conversion | An actor can implement `as_inference()` to return an exportable owned copy. |

Sampling must retain gradients from the action into policy parameters. Critic evaluation must
retain gradients with respect to its action input even when critic parameters are frozen.
`training` controls each forward's normalization statistics. The algorithm supplies the update
schedule; the models implement their own normalization and architecture.

The runner requires positive `num_steps_per_env`, `updates_per_step`, `batch_size`,
`warmup_transitions`, `replay_capacity`, `replay_max_bytes`, `n_step` and `save_interval`.
Batch size must be at least two and warmup cannot exceed replay capacity.

The environment supplies the RSL-RL `VecEnv` interface and transition metadata below.
Plain heads consume concatenated vector groups. `critic` may use privileged groups.
One iteration collects `num_steps_per_env` vector steps. Each eligible step runs
`updates_per_step` learner updates after warmup; each update samples `batch_size` transitions.
Actor and temperature update every `actor_update_period` critic updates (default 2).
Replay capacity and warmup count individual transitions, not vector steps.
Memory limits include ring storage and pending n-step transitions.

Policy actions and replay actions share the same coordinates. The tanh policy has support
`[-action_support, action_support]` (default 1), including the log-density scaling correction.
Disable wrapper action clipping. The task may map these actions to actuator commands as usual.

## Episode boundaries

Keep the four-result `step()` API: `(observations, rewards, dones, extras)`.
Supply boolean `[num_envs]` tensors `extras["terminated"]` and `extras["truncated"]`,
with `dones == terminated | truncated`. On a reset, also supply:

- `extras["final_obs"]`: all observation groups after the action, **before reset**,
  with batch size `num_envs`.
- `extras["final_obs_mask"]`: boolean `[num_envs]`, true for each valid final row.

Returned reset observations begin the next trajectory. Replay clones its inputs and uses
final observations for the ended transition. A true termination masks bootstrap; a time-limit
truncation stops n-step accumulation but bootstraps from the final observation. If both flags
are true, termination wins. Finite-horizon tasks must mark their horizon as terminal;
an adapter must mark finite-horizon endings accordingly.
The runner rejects missing final rows. It cannot independently detect incorrectly supplied
metadata from another environment adapter. The IsaacLab integration
enables pre-reset capture and preserves separate boundary flags for the off-policy runner.

## Symmetry extension

FlashSAC accepts the same `symmetry_cfg` fields and task callback as PPO through
`rsl_rl.extensions.Symmetry`: `data_augmentation_func`, `use_data_augmentation`,
`use_mirror_loss` and `mirror_loss_coeff`. Construction supplies the environment to the callback.

Replay augmentation pairs current observations, actual n-step successors and actions in one
callback invocation. Original rows precede transformed rows. Rewards, boundary flags and actual
horizon discounts are repeated unchanged, so their invariance is a task assumption. Every raw
observation group, including auxiliary labels, must have a transform. Replay storage and collection-time
reward statistics are not modified. `batch_size` counts original draws before augmentation.

Optional mirror loss compares deterministic tanh actions on transformed observations with detached
transforms of the original policy actions. It is applied only on actor update steps, without changing
normalization statistics. With only augmentation enabled it is logged but does not contribute gradients.
Agile's G1 recipe uses paired augmentation without a mirror-loss penalty.

## Checkpoints and inference

Format version 1 includes online/target models, optimizer/scheduler/scaler state,
reward statistics, action support, exploration state, counters and RNG state.
Restoring optimizer state validates the training settings.
Runner restoration starts fresh replay and environment trajectories and reports that fact;
exact continuation and conversion from Agile checkpoints are not implemented.

`get_inference_policy()` returns an owned deterministic actor with a no-op `reset()`.
JIT and ONNX exports accept flattened actor observation groups in `policy.obs_groups` order.
Architecture-specific export preparation belongs to the supplied actor, not the core runner.
Replay allocation is deferred until collection, so loading for play/export does not allocate it.

## Supported scope

The initial implementation supports eager feed-forward training on one CPU or CUDA device,
including explicit observation histories and optional CUDA FP16 AMP.
Distributed training and `torch.compile` are rejected. Compositional extensions supply models
and auxiliary hooks while inheriting the same update implementation.

Task curricula, reward relabeling and the full G1/One-S recipe remain in Agile. Reference parity is
tested with the compositional architecture. Short native rigid G1 training with symmetry and forced
timeouts, checkpoint playback, JIT/ONNX and LEAPP export have passed. Locomotion learning curves,
throughput equivalence and hardware performance have not been validated.

## Learning-parameter curricula

`algorithm.curriculum_cfg` maps parameter names to dictionaries with `type` and `params`.
The generic `ParameterScheduler` in the existing utils module supports:

- `linear`: `start_step`, `end_step`, `start_value`, `end_value`. Values are held before/after
  the endpoints and interpolated between them.
- `discrete`: increasing integer `steps` starting at zero and a matching list of `values`.
  A new value applies at its milestone, inclusively.

All steps count **completed vectorized collection steps**, including warmup. The runner applies
schedules before collecting the next transition and keeps those values fixed through the associated
updates. Values are logged under `curriculum/<parameter>`. Replay actions remain in executed coordinates.

FlashSAC currently supports `action_support`, `learning_rate`, `target_tau`,
`actor_update_period` (discrete integer schedules only), and `temperature_target_sigma`.
An explicit learning-rate curriculum replaces cosine decay for the actor, critic and temperature
optimizers. The temperature target's derived entropy is updated when its sigma changes. Unknown or
non-runtime parameters, including `gamma` and `num_bins`, fail at initialization; replay discounts and
network shapes cannot be changed by assigning a scalar attribute. The scheduling utility is generic;
this integration currently belongs to FlashSAC/OffPolicyRunner, not PPO.

The algorithm checkpoint stores the applied schedule step and the runner stores completed collection
steps. A full resume restores current values, then advances at the next collection. Inference and
export retain the saved action bound. Schedule configuration must match for optimizer resume;
weights-only transfer may use a new schedule. With no curriculum configured, existing behavior is unchanged.

## Replay curricula and time budgets

`OffPolicyRunner` accepts optional `replay_cfg`, `max_time_s`, and
`save_interval_updates` settings. `replay_cfg.class_name` resolves a transform
constructed with the environment and remaining configuration arguments. The
transform starts with the remaining wall-time budget, runs before each collection,
adds transition fields, and transforms sampled batches. Its `sum_keys` identifies
fields that replay must accumulate with the same n-step discount and episode
boundaries as rewards. This supports current-weight reward relabeling without
putting task-specific reward logic into the algorithm.

`max_time_s` stops collection and updates at the deadline and still saves the final
learner. `learn(process_started=...)` can include environment setup in the budget;
otherwise the clock starts at `learn`. `save_interval_updates` overrides periodic
iteration-based saves and counts learner updates. These options default to disabled.


## Distributed training

Launch one process per GPU with `torchrun`. `OffPolicyRunner` uses `WORLD_SIZE`,
`RANK` and `LOCAL_RANK`, initializes NCCL, and requires each process to use its
local CUDA device. CPU/Gloo is supported for distributed contract tests.

Every rank owns its environments and replay, but trains the same policy: online
and target models are broadcast initially, and actor, critic, temperature and
compositional auxiliary gradients are averaged before optimizer steps. Scaled AMP
gradients are reduced before unscaling so overflow/skip decisions agree.
Reward normalization combines return moments and maxima across all ranks.

Environment count, `batch_size`, `warmup_transitions`, `replay_capacity`, and
`replay_max_bytes` are **per rank**. Thus the effective minibatch is
`world_size * batch_size` before symmetry augmentation. `updates_per_step` and
curriculum steps count synchronized vector collections, not total transitions.
Updates start only after every local replay has warmed up. Keep the algorithm
configuration and iteration count identical across ranks. Wall-time stopping is
collective: if any rank reaches its deadline, all ranks stop together.

Batch normalization uses local minibatch statistics. Floating running buffers are
averaged after updates, and integer counters take the maximum, so inference and
checkpoints use identical state on all ranks. This is not global-batch SyncBatchNorm.
Only rank zero writes logs and checkpoints; learner metrics are averaged across
ranks, while episode metrics follow RSL-RL's rank-zero logging convention.
Resume restores learner state with fresh local replay and rank-specific random
streams, as with single-GPU fresh-replay resume. Time-based environment curricula
must synchronize their own clocks; the Agile task's replay extension does so.
