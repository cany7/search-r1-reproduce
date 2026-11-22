# Adapted from tinker-cookbook/rl/train.py (configuration fields) at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
from pathlib import Path
from typing import Any, Literal

import chz

from search_r1.env.search_env import SearchR1DatasetBuilder
from search_r1.env.tools import RetrievalConfig
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
    parent_run: str | None = None
    parent_step: int | None = None
    max_steps: int | None = None
    sampling_mode: Literal["ordinary", "dynamic"] = "ordinary"
    max_sampling_rounds: int = 8

    skyrl_compat: bool = True

    def validate(self):
        """Reject unsupported framework options before any resource is created."""
        if (self.parent_run is None) != (self.parent_step is None):
            raise ValueError("Specify parent_run and parent_step together")
        if self.parent_run is not None and not self.parent_run.strip():
            raise ValueError("parent_run must name a directory")
        if self.load_checkpoint_path is not None and not self.load_checkpoint_path.strip():
            raise ValueError("load_checkpoint_path must name a weights checkpoint")
        if self.parent_step is not None and (
            type(self.parent_step) is not int or self.parent_step < 0 or self.load_checkpoint_path
        ):
            raise ValueError(
                "A fork needs a nonnegative parent_step and no weights-only initializer"
            )
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


def from_snapshot(snapshot):
    """Rebuild the saved recipe, using defaults for fields absent from the snapshot."""
    builder = dict(snapshot["dataset_builder"])
    builder["retrieval_config"] = RetrievalConfig(**builder.get("retrieval_config", {}))
    values = {k: v for k, v in snapshot.items() if k in chz.chz_fields(Config)}
    values["dataset_builder"] = SearchR1DatasetBuilder(**builder)
    if values.get("stream_minibatch_config") is not None:
        values["stream_minibatch_config"] = StreamMinibatchConfig(
            **values["stream_minibatch_config"]
        )
    if "sampling_mode" not in values and snapshot.get("parent_state_path"):
        values["sampling_mode"] = "dynamic"
    return Config(**values)
