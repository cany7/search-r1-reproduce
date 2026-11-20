"""CLI for Search-R1 replication."""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from search_r1.config import CLIConfig
from search_r1.env import renderers as model_info
from search_r1.env.search_env import SearchR1DatasetBuilder
from search_r1.env.tools import RetrievalConfig
from search_r1.training import runtime
from search_r1.training.config import Config as TrainingConfig
from search_r1.training.config import StreamMinibatchConfig


def build_config(cli_config: CLIConfig) -> TrainingConfig:

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
        limit_search=cli_config.limit_search,
        max_turns_unanswered_reward=cli_config.max_turns_unanswered_reward,
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
        stream_minibatch_config = StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )
        bs_str = f"bs{cli_config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{cli_config.batch_size}"

    # Build run name
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    run_name = (
        f"search_r1_{cli_config.sampling_mode}_{model_name_short}_{bs_str}_gs{cli_config.group_size}_"
        f"seed{cli_config.seed}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = str(
            Path(os.environ.get("TINKER_COOKBOOK_RUNS_DIR") or "~/tinker-runs").expanduser()
            / "rl_search"
            / run_name
        )

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Build training config
    config = TrainingConfig(
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
        parent_run=cli_config.parent_run,
        parent_step=cli_config.parent_step,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        skyrl_compat=cli_config.skyrl_compat,
        num_substeps=cli_config.num_substeps,
        sampling_mode=cli_config.sampling_mode,
        max_sampling_rounds=cli_config.max_sampling_rounds,
        remove_constant_reward_groups=cli_config.sampling_mode == "dynamic",
    )

    config.validate()
    return config


async def cli_main(cli_config: CLIConfig) -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    config = build_config(cli_config)
    await runtime.main(config, behavior=cli_config.behavior_if_log_dir_exists)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
