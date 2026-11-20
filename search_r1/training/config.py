# Adapted from tinker-cookbook/rl/train.py (configuration fields) at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
from pathlib import Path
from typing import Any, Literal

import chz

from search_r1.env.types import RLDatasetBuilder


@chz.chz
class StreamMinibatchConfig:
    groups_per_batch: int
    num_minibatches: int


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder
    model_name: str
    recipe_name: str
    max_tokens: int
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    eval_every: int = 20
    save_every: int = 20
    evaluator_builders: list[Any] = chz.field(default_factory=list)
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    session_name: str | None = None
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    kl_reference_config: Any | None = None
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    num_substeps: int = 1
    lora_rank: int = 32
    temperature: float = 1.0
    compute_post_kl: bool = False
    remove_constant_reward_groups: bool = False
    rollout_error_tolerance: bool | Any | None = None
    termination: Any | None = None
    enable_trace: bool = False
    span_chart_every: int = 0
    async_config: Any | None = None
    stream_minibatch_config: StreamMinibatchConfig | None = None
    base_url: str | None = None
    ttl_seconds: int | None = 604800
    rolling_save_every: int = 0
    rolling_ttl_seconds: int = 7200
    num_groups_to_log: int = 4
    rollout_json_export: bool = True
    num_epochs: int = 1
    max_steps: int | None = None
    sampling_mode: Literal["ordinary", "dynamic"] = "ordinary"
    max_sampling_rounds: int = 8

    def validate(self):
        """Reject unsupported framework options before any resource is created."""
        if self.async_config is not None:
            raise ValueError("async_config is unsupported; use synchronous sampling")
        for name in (
            "kl_penalty_coef",
            "kl_discount_factor",
            "kl_reference_config",
            "compute_post_kl",
            "termination",
            "enable_trace",
            "span_chart_every",
            "evaluator_builders",
            "rolling_save_every",
        ):
            if getattr(self, name):
                raise ValueError(f"{name} is not supported by the Search-R1 loop")
        if self.rollout_error_tolerance not in (None, False):
            raise ValueError("Only fail-fast rollouts are supported")
        if self.loss_fn not in ("importance_sampling", "ppo", "cispo"):
            raise ValueError(f"Unsupported loss_fn: {self.loss_fn}")
        if self.num_substeps < 1 or self.num_epochs < 1 or self.max_sampling_rounds < 1:
            raise ValueError("substeps, epochs and sampling rounds must be positive")
        if self.max_tokens < 1 or self.save_every < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid token, save or optimizer configuration")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        size = self.dataset_builder.batch_size
        if size < 1 or self.dataset_builder.group_size < 1:
            raise ValueError("batch_size and group_size must be positive")
        if self.sampling_mode not in ("ordinary", "dynamic"):
            raise ValueError("Unknown sampling_mode")
        if self.sampling_mode == "dynamic":
            if self.stream_minibatch_config is not None or self.num_substeps != 1:
                raise ValueError("Dynamic sampling does not support streaming or multiple substeps")
            if self.num_epochs != 1:
                raise ValueError("Dynamic sampling consumes one shuffled dataset")
        stream = self.stream_minibatch_config
        if stream is not None:
            if stream.groups_per_batch != size or stream.num_minibatches < 1:
                raise ValueError(
                    "Streaming groups must match batch_size; minibatches must be positive"
                )
            if size % (self.num_substeps * stream.num_minibatches):
                raise ValueError("batch_size must be divisible by substeps * num_minibatches")
