# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Off-policy collection and update scheduling using the RSL-RL environment and logger APIs."""

from __future__ import annotations

import copy
import math
import os
import time
import torch
import warnings
from math import prod as math_prod
from pathlib import Path
from tensordict import TensorDict
from torch import nn

from rsl_rl.env import VecEnv
from rsl_rl.storage import ReplayStorage
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class OffPolicyRunner:
    """Collect vector steps and train FlashSAC from bounded replay.

    An optional ``replay_cfg`` resolves an environment-side transform with
    ``start(remaining_seconds)``, ``before_collection(step)``,
    ``process_transition(transition)`` and ``process_batch(batch)`` methods.
    Its ``sum_keys`` declares fields accumulated over the replay's n-step horizon.
    This keeps simulator-specific reward curricula outside the learner.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the learner; defer replay allocation until training collects its first step."""
        if int(os.getenv("WORLD_SIZE", "1")) > 1 or train_cfg.get("multi_gpu"):
            raise ValueError("OffPolicyRunner currently supports one learner device.")
        self.env = env
        self.cfg = copy.deepcopy(train_cfg)
        self.device = device
        for key in (
            "num_steps_per_env",
            "updates_per_step",
            "batch_size",
            "warmup_transitions",
            "replay_capacity",
            "replay_max_bytes",
            "n_step",
            "save_interval",
        ):
            if type(self.cfg[key]) is not int or self.cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.cfg["batch_size"] < 2 or self.cfg["warmup_transitions"] > self.cfg["replay_capacity"]:
            raise ValueError("FlashSAC needs batch_size >= 2 and enough replay capacity for warmup.")
        if getattr(env, "clip_actions", None) is not None:
            raise ValueError("FlashSAC requires unclipped wrapper actions; configure action_support in the algorithm.")
        obs = self.env.get_observations().to(device)
        self.obs_shapes = {key: tuple(value.shape[1:]) for key, value in obs.items()}
        cls = resolve_callable(self.cfg["algorithm"]["class_name"])
        self.alg = cls.construct_algorithm(obs, env, self.cfg, device)
        self.logger = Logger(log_dir, self.cfg, env.cfg, env.num_envs, False, 1, 0, device)
        self.current_learning_iteration = 0
        self.collection_steps = 0
        self.replay = None
        self.max_time_s = self.cfg.get("max_time_s")
        if self.max_time_s is not None and (not math.isfinite(self.max_time_s) or self.max_time_s <= 0):
            raise ValueError("max_time_s must be finite and positive.")
        self.save_interval_updates = self.cfg.get("save_interval_updates")
        if self.save_interval_updates is not None and (
            type(self.save_interval_updates) is not int or self.save_interval_updates < 1
        ):
            raise ValueError("save_interval_updates must be a positive integer.")
        replay_cfg = copy.deepcopy(self.cfg.get("replay_cfg"))
        self.replay_transform = None
        if replay_cfg:
            self.replay_transform = resolve_callable(replay_cfg.pop("class_name"))(env, **replay_cfg)

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False, *, process_started: float | None = None
    ) -> None:
        """Collect num_steps_per_env vector steps per iteration, with updates after every step."""
        started = time.perf_counter() if process_started is None else process_started
        deadline = None if self.max_time_s is None else started + self.max_time_s
        if deadline is not None and time.perf_counter() >= deadline:
            raise RuntimeError("Setup exhausted the training time budget.")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.alg.train_mode()
        self.logger.init_logging_writer()
        obs = self.env.get_observations().to(self.device).clone()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        if self.replay_transform is not None:
            self.replay_transform.start(None if deadline is None else deadline - time.perf_counter())
        for it in range(start_it, total_it):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            collect_time = learn_time = 0.0
            totals, counts = {}, {}
            for _ in range(self.cfg["num_steps_per_env"]):
                start = time.perf_counter()
                if deadline is not None and start >= deadline:
                    break
                if self.replay_transform is not None:
                    self.replay_transform.before_collection(self.collection_steps)
                for name, value in self.alg.update_curriculum(self.collection_steps).items():
                    key = f"curriculum/{name}"
                    totals[key] = totals.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1
                with torch.no_grad():
                    if self.replay is None or self.replay.size < self.cfg["warmup_transitions"]:
                        actions = (
                            torch.rand(self.env.num_envs, self.env.num_actions, device=self.device) * 2 - 1
                        ) * self.alg.action_support
                    else:
                        actions = self.alg.act(obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_obs, rewards, dones = next_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    if not {"terminated", "truncated"} <= extras.keys():
                        raise ValueError(
                            "Off-policy collection requires separate terminated and truncated flags in extras."
                        )
                    terminated = extras["terminated"].to(self.device).bool()
                    truncated = extras["truncated"].to(self.device).bool()
                    if not torch.equal(dones.bool(), terminated | truncated):
                        raise ValueError("dones must equal terminated | truncated.")
                    successors = next_obs.clone()
                    if dones.any():
                        if "final_obs" not in extras or "final_obs_mask" not in extras:
                            raise ValueError("Off-policy collection requires final_obs and final_obs_mask at reset.")
                        mask = extras["final_obs_mask"].to(self.device).bool()
                        if not torch.all(mask[dones.bool()]):
                            raise ValueError("Missing final observations for reset environments.")
                        final = TensorDict(extras["final_obs"], batch_size=[self.env.num_envs]).to(self.device)
                        successors[dones.bool()] = final[dones.bool()]
                    transition = TensorDict(
                        dict(
                            observations=obs,
                            next_observations=successors,
                            actions=actions,
                            rewards=rewards,
                            terminated=terminated,
                            truncated=truncated,
                        ),
                        batch_size=[self.env.num_envs],
                    )
                    if self.replay_transform is not None:
                        transition = self.replay_transform.process_transition(transition)
                    if self.replay is None:
                        self.replay = ReplayStorage(
                            transition,
                            self.cfg["replay_capacity"],
                            self.cfg["replay_max_bytes"],
                            self.alg.gamma,
                            self.cfg["n_step"],
                            sum_keys=("rewards",) if self.replay_transform is None else self.replay_transform.sum_keys,
                        )
                    self.replay.add(transition)
                    self.alg.observe_rewards(rewards, terminated, truncated)
                    self.logger.process_env_step(rewards, dones, extras, None)
                    obs = next_obs.clone()
                    self.collection_steps += 1
                collect_time += time.perf_counter() - start
                start = time.perf_counter()
                if self.replay.size >= self.cfg["warmup_transitions"]:
                    for _ in range(self.cfg["updates_per_step"]):
                        if deadline is not None and time.perf_counter() >= deadline:
                            break
                        batch = self.replay.sample(self.cfg["batch_size"])
                        if self.replay_transform is not None:
                            batch = self.replay_transform.process_batch(batch)
                        metrics = self.alg.update(batch)
                        for key, value in metrics.items():
                            if not torch.isfinite(value).all():
                                raise RuntimeError(f"Nonfinite FlashSAC metric: {key}")
                            totals[key] = totals.get(key, 0.0) + value.item()
                            counts[key] = counts.get(key, 0) + 1
                        if (
                            self.logger.writer is not None
                            and self.save_interval_updates is not None
                            and self.alg.update_step % self.save_interval_updates == 0
                        ):
                            self.save(os.path.join(self.logger.log_dir, f"model_update_{self.alg.update_step}.pt"))
                learn_time += time.perf_counter() - start
            self.current_learning_iteration = it + 1
            with torch.no_grad():
                _, std = self.alg.actor.get_mean_and_std(self.alg.actor_observations(obs), training=False)
            self.logger.log(
                it,
                start_it,
                total_it,
                collect_time,
                learn_time,
                {key: value / counts[key] for key, value in totals.items()},
                self.alg.learning_rate,
                std,
                None,
                remaining_time=None if deadline is None else max(0.0, deadline - time.perf_counter()),
            )
            if (
                self.logger.writer is not None
                and self.save_interval_updates is None
                and (it + 1) % self.cfg["save_interval"] == 0
            ):
                self.save(os.path.join(self.logger.log_dir, f"model_{it + 1}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save learner state; replay and simulator continuation are not included."""
        state = self.alg.save()
        state.update(iter=self.current_learning_iteration, collection_steps=self.collection_steps, infos=infos)
        torch.save(state, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load learner state with fresh replay and fresh environment trajectories."""
        state = torch.load(path, weights_only=False, map_location=map_location or self.device)
        if self.alg.load(state, load_cfg, strict):
            self.current_learning_iteration = state["iter"]
            self.collection_steps = state["collection_steps"]
        self.replay = None
        self.alg.reward_return = torch.zeros(1, device=self.device)
        self.alg.noise_count.zero_()
        warnings.warn(
            "FlashSAC restored with fresh replay and environment trajectories; this is not exact continuation.",
            stacklevel=2,
        )
        return state["infos"]

    def get_inference_policy(self, device: str | None = None) -> nn.Module:
        """Return deterministic actions for observation dictionaries."""
        policy = copy.deepcopy(self.alg.get_policy()).eval().to(device or self.device)
        return policy

    def _export_model(self) -> tuple[nn.Module, torch.Tensor]:
        policy = self.get_inference_policy("cpu")
        shapes = {key: self.obs_shapes[key] for key in policy.obs_groups}
        model = _FlatPolicy(policy, shapes).eval()
        return model, torch.zeros(1, sum(math_prod(shape) for shape in shapes.values()))

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the deterministic actor with flattened actor observations."""
        model, example = self._export_model()
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.jit.trace(model, example).save(str(Path(path) / filename))

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the actor without allocating replay or including privileged critic inputs."""
        model, example = self._export_model()
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            (example,),
            str(Path(path) / filename),
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=18,
            dynamo=False,
            verbose=verbose,
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register source provenance with the shared RSL-RL logger."""
        self.logger.git_status_repos.append(repo_file_path)


class _FlatPolicy(nn.Module):
    def __init__(self, policy: nn.Module, shapes: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        self.policy = policy
        self.shapes = shapes

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        values = {}
        offset = 0
        for name, shape in self.shapes.items():
            size = math_prod(shape)
            values[name] = obs[:, offset : offset + size].reshape(obs.shape[0], *shape)
            offset += size
        return self.policy(TensorDict(values, batch_size=[obs.shape[0]]))
