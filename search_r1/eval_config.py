"""Map the saved cookbook training configuration to offline evaluation settings."""

from typing import Any


def eval_settings_from_training_config(training: dict[str, Any]) -> dict[str, Any]:
    """Use resolved training values, including defaults, rather than replaying CLI overrides.

    Evaluation data selection, output paths and concurrency remain evaluation-specific.
    The schema is the config.json written by the pinned cookbook and SearchR1DatasetBuilder.
    Missing fields fail explicitly instead of silently falling back to evaluation defaults.
    """
    builder = training["dataset_builder"]
    return {
        "base_model": training["model_name"],
        "lora_rank": training["lora_rank"],
        "renderer_name": training["renderer_name"],
        "max_tokens": training["max_tokens"],
        "temperature": training["temperature"],
        "prompt_name": builder["prompt_name"],
        "answer_score": builder["answer_score"],
        "format_coef": builder["format_coef"],
        "no_search_coef": builder["no_search_coef"],
        "max_turns": builder["max_turns"],
        "max_trajectory_tokens": builder["max_trajectory_tokens"],
        "max_generation_tokens": builder["max_generation_tokens"],
        "context_overflow_reward": builder["context_overflow_reward"],
        "retrieval_url": builder["retrieval_url"],
        "n_results": builder["retrieval_config"]["n_results"],
    }
