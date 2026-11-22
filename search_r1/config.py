"""Public chz options shared by ordinary and dynamic Search-R1 CLIs."""

from typing import Literal

import chz

from search_r1.env.renderers import DEFAULT_RENDERER_NAME
from search_r1.env.search_env import DEFAULT_RETRIEVAL_URL, PromptName


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    renderer_name: str | None = DEFAULT_RENDERER_NAME

    # Training parameters
    learning_rate: float = 4e-5
    batch_size: int = 512
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0
    save_every: int = 25

    # Dataset parameters
    group_size: int = 8
    max_turns: int = 5
    format_coef: float = 0.1
    max_trajectory_tokens: int = 32 * 1024

    # Prompt and reward. Defaults: fast-thinking prompt and F1+ reward.
    # The recipe's originals are prompt_name=original answer_score=em no_search_coef=0.0.
    prompt_name: PromptName = "fast"
    answer_score: Literal["f1", "em"] = "f1"
    no_search_coef: float = 0.1

    # BM25 retrieval server
    retrieval_url: str = DEFAULT_RETRIEVAL_URL
    n_results: int = 3
    limit_search: bool = chz.field(
        default=False, doc="Run only the first search per turn and its first three queries."
    )
    max_turns_unanswered_reward: float | None = chz.field(
        default=None,
        doc="Training reward for an episode that reaches max_turns without a valid answer "
        "(requires limit_search). None keeps the shared scorer's reward.",
    )

    # Streaming configuration
    stream_minibatch: bool = False
    num_minibatches: int = 4

    num_substeps: int = 1
    sampling_mode: Literal["ordinary", "dynamic"] = "ordinary"
    max_sampling_rounds: int = 8

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = "search-r1-reproduce-experiment"
    wandb_name: str | None = None
    # Display name for the Tinker session; defaults to wandb_name, or the
    # generated run name when wandb_name is unset.
    session_name: str | None = None

    behavior_if_log_dir_exists: Literal["ask", "delete", "resume", "raise"] = "ask"

    parent_run: str | None = None
    parent_step: int | None = None
    max_steps: int | None = None

    load_checkpoint_path: str | None = None  # Initialize weights only, with a fresh optimizer.

    # Resume via create-LoRA-client + load_state, which the self-hosted SkyRL server
    # (tinker 0.27 protocol) accepts. See backends/tinker_backend.py.
    skyrl_compat: bool = True
