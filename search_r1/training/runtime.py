"""Own run initialization, replay and SDK lifetime for both training CLIs."""

import logging
from pathlib import Path

import tinker

from search_r1.backends import tinker_backend
from search_r1.training.checkpoints import (
    CheckpointManager,
    prepare_run,
    run_lock,
    separate_directories,
)
from search_r1.training.logging import RunLogger, rollback_logs
from search_r1.training.loop import run_updates
from search_r1.training.usage import CostTracker, TokenUsage, fetch_model_pricing


async def close_dataset(dataset):
    tools = {
        id(b.search_tool): b.search_tool
        for b in dataset.env_group_builders
        if hasattr(b, "search_tool")
    }
    for tool in tools.values():
        await tool.aclose()


async def main(config, *, behavior="resume"):
    config.validate()
    directory = Path(config.log_path).resolve()
    if config.parent_run:
        separate_directories(Path(config.parent_run).expanduser().resolve(), directory)
    with run_lock(directory, behavior):
        dataset, _ = await config.dataset_builder()
        try:
            if not len(dataset):
                raise ValueError("Dataset does not contain a full batch")
            start = prepare_run(config, dataset)
            end = len(dataset) * config.num_epochs
            if config.sampling_mode == "dynamic":
                end = (
                    start.step
                    + (len(dataset.env_group_builders) - start.offset) // dataset.batch_size
                )
            if config.max_steps is not None:
                end = (
                    config.max_steps
                    if config.sampling_mode == "dynamic"
                    else min(end, config.max_steps)
                )
            if start.step >= end:
                return
            rollback_logs(directory, start.step)
            log = RunLogger(config, fork="parent_state_path" in start.fields)
            try:
                log.message(
                    f"Starting at update {start.step}, next prompt {start.offset}, target {end}"
                )
                await _train(config, dataset, start, end, log)
            except BaseException:
                log.close(exit_code=1)
                raise
            else:
                log.close()
        finally:
            await close_dataset(dataset)


async def _train(config, dataset, start, end, log):
    costs = CostTracker(
        fetch_model_pricing(config.model_name),
        cumulative=TokenUsage(**start.record.get("cumulative_tokens", {})),
        discarded=TokenUsage(**start.record.get("discarded_tokens", {})),
    )
    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        display_name=config.session_name or config.wandb_name,
        user_metadata={"recipe_name": config.recipe_name},
    )
    with tinker_backend.training_session(
        service_client, skyrl_compat=config.skyrl_compat
    ) as service:
        metadata = {}
        if log.run is not None and log.run.url:
            metadata["wandb_link"] = log.run.url
        if config.renderer_name:
            metadata["renderer_name"] = config.renderer_name
        path = start.record.get("state_path") or start.weights_path
        if path:
            renderer = await tinker_backend.get_renderer_name_from_checkpoint_async(service, path)
            if renderer and renderer != config.renderer_name:
                logging.getLogger(__name__).warning(
                    "Checkpoint renderer %s differs from %s", renderer, config.renderer_name
                )
            client = await tinker_backend.load_training_client(
                service,
                path,
                config.model_name,
                config.lora_rank,
                with_optimizer=bool(start.record),
                skyrl_compat=config.skyrl_compat,
                user_metadata=metadata,
            )
        else:
            client = await service.create_lora_training_client(
                config.model_name, rank=config.lora_rank, user_metadata=metadata
            )
        manager = CheckpointManager(client, Path(config.log_path), config.ttl_seconds)
        await run_updates(
            config,
            dataset,
            start.step,
            end,
            start.offset,
            client,
            manager,
            log,
            costs,
            checkpoint_fields=start.fields,
        )
