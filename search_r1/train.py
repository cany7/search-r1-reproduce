"""CLI for Search-R1 replication."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from search_r1.search_env import DEFAULT_RETRIEVAL_URL, PromptName, SearchR1DatasetBuilder
from search_r1.tools import RetrievalConfig


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    renderer_name: str | None = "qwen3_5_disable_thinking"

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

    # Streaming configuration
    stream_minibatch: bool = False
    num_minibatches: int = 4

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = "search-r1-reproduce"
    wandb_name: str | None = None
    # Display name for the Tinker session; defaults to wandb_name, or the
    # generated run name when wandb_name is unset.
    session_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    # Build retrieval config
    retrieval_config = RetrievalConfig(n_results=cli_config.n_results)

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    builder = SearchR1DatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        retrieval_url=cli_config.retrieval_url,
        retrieval_config=retrieval_config,
        seed=cli_config.seed,
        max_turns=cli_config.max_turns,
        format_coef=cli_config.format_coef,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        prompt_name=cli_config.prompt_name,
        answer_score=cli_config.answer_score,
        no_search_coef=cli_config.no_search_coef,
    )

    # Configure streaming minibatch
    if cli_config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )
        bs_str = f"bs{cli_config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{cli_config.batch_size}"

    # Build run name
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"search_r1_{model_name_short}_{bs_str}_gs{cli_config.group_size}_"
        f"seed{cli_config.seed}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = cli_utils.runs_path("rl_search", run_name)

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Build training config
    config = train.Config(
        model_name=cli_config.model_name,
        recipe_name="recipe_search_tool",
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        session_name=cli_config.session_name or wandb_name,
        lora_rank=cli_config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
        max_steps=cli_config.max_steps,
    )

    # Run training
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
