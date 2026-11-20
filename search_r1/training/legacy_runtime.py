"""Temporary startup/checkpoint/log adapter, to be replaced in refactor stage 5.

No rollout, advantage calculation or optimizer update is delegated to cookbook.
Imports are deliberately confined here so computation and CLI help stay independent.
"""

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.cost_estimation import fetch_model_pricing

from search_r1.training.loop import run_updates
from search_r1.training.usage import CostTracker, TokenUsage
from search_r1.wandb_compat import persistent_wandb_run


async def close_dataset(dataset):
    """The Search-R1 dataset owns one shared retrieval client."""
    tools = {
        id(b.search_tool): b.search_tool
        for b in dataset.env_group_builders
        if hasattr(b, "search_tool")
    }
    for tool in tools.values():
        await tool.aclose()


async def main(config):
    config.validate()
    with persistent_wandb_run():
        log = ml_log.setup_logging(
            log_dir=config.log_path,
            wandb_project=config.wandb_project,
            config=config,
            wandb_name=config.wandb_name,
        )
    try:
        resume = checkpoint_utils.get_last_checkpoint(config.log_path)
        step = resume.batch if resume else 0
        if resume and log.store is not None:
            log.store.truncate_metrics(step)
        if (
            config.sampling_mode == "dynamic"
            and resume
            and type(resume.get("next_prompt_offset")) is not int
        ):
            raise ValueError("Dynamic checkpoint lacks next_prompt_offset")
        costs = CostTracker(
            fetch_model_pricing(config.model_name),
            cumulative=TokenUsage(**(resume.get("cumulative_tokens", {}) if resume else {})),
            discarded=TokenUsage(**(resume.get("discarded_tokens", {}) if resume else {})),
        )
        dataset, _ = await config.dataset_builder()
        try:
            if not len(dataset):
                raise ValueError("Dataset does not contain a full batch")
            end = len(dataset) * config.num_epochs
            if config.max_steps is not None:
                end = min(end, config.max_steps)
            if step >= end:
                return
            offset = (
                resume.get("next_prompt_offset", (step % len(dataset)) * dataset.batch_size)
                if resume
                else 0
            )
            with tinker.ServiceClient(
                base_url=config.base_url,
                display_name=config.session_name or config.wandb_name,
                user_metadata={"recipe_name": config.recipe_name},
            ) as service:
                metadata = {}
                if url := log.get_logger_url():
                    metadata["wandb_link"] = url
                checkpoint_utils.add_renderer_name_to_user_metadata(metadata, config.renderer_name)
                state_path = resume.state_path if resume else config.load_checkpoint_path
                if state_path:
                    await checkpoint_utils.check_renderer_name_for_checkpoint_async(
                        service, state_path, config.renderer_name
                    )
                    create = (
                        service.create_training_client_from_state_with_optimizer
                        if resume
                        else service.create_training_client_from_state
                    )
                    client = await create(state_path, user_metadata=metadata)
                else:
                    client = await service.create_lora_training_client(
                        config.model_name, rank=config.lora_rank, user_metadata=metadata
                    )
                manager = checkpoint_utils.CheckpointManager(
                    training_client=client,
                    service_client=service,
                    log_path=config.log_path,
                    save_every=0,
                    ttl_seconds=config.ttl_seconds,
                    store=log.store,
                )
                await run_updates(config, dataset, step, end, offset, client, manager, log, costs)
                log.log_metrics(costs.cumulative_metrics(), step=end)
        finally:
            await close_dataset(dataset)
    finally:
        log.close()
