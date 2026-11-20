"""Stage three: synchronous refill from fresh prompts, one optimizer step per full batch."""

import argparse
import asyncio
import fcntl
import json
import logging
from pathlib import Path

import chz
import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.cost_estimation import fetch_model_pricing

from search_r1.env.search_env import SearchR1DatasetBuilder
from search_r1.env.tools import RetrievalConfig
from search_r1.skyrl_compat import patch_create_from_state
from search_r1.training.config import Config
from search_r1.training.legacy_runtime import close_dataset
from search_r1.training.loop import run_updates
from search_r1.training.usage import CostTracker, TokenUsage
from search_r1.wandb_compat import persistent_wandb_run

START_STEP = 100
logger = logging.getLogger(__name__)


@chz.chz
class Stage3Config(Config):
    parent_state_path: str
    sampling_mode: str = "dynamic"


def load_run(parent: Path, output: Path, target: int, save_every: int, max_rounds: int):
    """Read the parent state in place. Only child-owned checkpoints go in the child's index."""
    parent, output = parent.resolve(), output.resolve()
    if output.parent != parent.parent or output == parent:
        raise ValueError("Use a separate sibling run directory for stage three")
    if target <= START_STEP or save_every < 0 or max_rounds < 1:
        raise ValueError("Expected target > 100, save_every >= 0 and max_rounds >= 1")
    settings = json.loads((parent / "config.json").read_text())
    builder = dict(settings.pop("dataset_builder"))
    if (
        builder["batch_size"],
        builder["group_size"],
        builder["seed"],
        builder.get("limit_search"),
    ) != (64, 8, 2, True):
        raise ValueError("Expected the 64 x 8, seed 2, limited-search stage-two run")
    if (
        settings["loss_fn"] != "importance_sampling"
        or settings["num_substeps"] != 1
        or settings["kl_penalty_coef"]
        or settings["async_config"]
        or settings["stream_minibatch_config"]
        or settings["remove_constant_reward_groups"]
        or settings["learning_rate"] != 4e-5
    ):
        raise ValueError("Unexpected stage-two objective, learning rate or sampling configuration")
    root = checkpoint_utils.get_last_checkpoint(str(parent))
    if not root or root.batch != START_STEP or root.name != "final":
        raise ValueError("Stage two must have its final full checkpoint at step 100")
    builder["retrieval_config"] = RetrievalConfig(**builder["retrieval_config"])
    settings.update(
        dataset_builder=SearchR1DatasetBuilder(**builder),
        log_path=str(output),
        wandb_name=output.name,
        session_name=output.name,
        max_steps=target,
        save_every=save_every,
        eval_every=0,
        remove_constant_reward_groups=True,
        sampling_mode="dynamic",
        parent_state_path=root.state_path,
        max_sampling_rounds=max_rounds,
    )
    config = Stage3Config(**settings)
    previous_config = output / "config.json"
    if previous_config.exists():
        old = json.loads(previous_config.read_text())
        if (old.get("parent_state_path"), old.get("wandb_name")) != (root.state_path, output.name):
            raise ValueError("Output directory belongs to a different run")
    elif output.exists() and any(p.name != ".stage3.lock" for p in output.iterdir()):
        raise ValueError("Refusing to adopt a populated run directory")
    records = checkpoint_utils.load_checkpoints_file(str(output))
    for record in records:
        if (
            record.get("parent_state_path") != root.state_path
            or record.get("stage3_run") != output.name
            or not record.state_path
            or record.batch is None
            or record.batch <= START_STEP
            or type(record.get("next_prompt_offset")) is not int
            or record.get("next_prompt_offset") < record.batch * builder["batch_size"]
        ):
            raise ValueError("Unexpected or incomplete child checkpoint record")
    resume = records[-1] if records else root
    if resume.batch > target:
        raise ValueError(f"Already at update {resume.batch}, beyond requested {target}")
    offset = resume.get("next_prompt_offset", resume.batch * builder["batch_size"])
    return config, resume, offset


async def train_updates(config, resume, offset, client, manager, log, costs):
    config.validate()
    dataset, _ = await config.dataset_builder()
    try:
        await run_updates(
            config,
            dataset,
            resume.batch,
            config.max_steps,
            offset,
            client,
            manager,
            log,
            costs,
            checkpoint_fields={
                "stage3_run": config.wandb_name,
                "parent_state_path": config.parent_state_path,
            },
        )
    finally:
        await close_dataset(dataset)


async def run(config, resume, offset):
    config.validate()
    if resume.batch == config.max_steps:
        return
    patch_create_from_state(config.model_name, config.lora_rank)
    with persistent_wandb_run():
        log = ml_log.setup_logging(
            log_dir=config.log_path,
            wandb_project=config.wandb_project,
            config=config,
            wandb_name=config.wandb_name,
        )
    try:
        if log.store is not None:
            log.store.truncate_metrics(resume.batch)
        costs = CostTracker(
            fetch_model_pricing(config.model_name),
            cumulative=TokenUsage(**resume.get("cumulative_tokens", {})),
            discarded=TokenUsage(**resume.get("discarded_tokens", {})),
        )
        with tinker.ServiceClient(
            base_url=config.base_url, display_name=config.session_name
        ) as service:
            await checkpoint_utils.check_renderer_name_for_checkpoint_async(
                service, resume.state_path, config.renderer_name
            )
            metadata = {}
            if url := log.get_logger_url():
                metadata["wandb_link"] = url
            checkpoint_utils.add_renderer_name_to_user_metadata(metadata, config.renderer_name)
            client = await service.create_training_client_from_state_with_optimizer(
                resume.state_path, user_metadata=metadata
            )
            logger.info(
                "Loaded full state %s at update %s, next prompt %d",
                resume.state_path,
                resume.batch,
                offset,
            )
            # Periodic/final saves are explicitly ordered after successful updates.
            manager = checkpoint_utils.CheckpointManager(
                training_client=client,
                service_client=service,
                log_path=config.log_path,
                save_every=0,
                ttl_seconds=config.ttl_seconds,
                store=log.store,
            )
            await train_updates(config, resume, offset, client, manager, log, costs)
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()
    if (
        args.output.resolve().parent != args.parent.resolve().parent
        or args.output.resolve() == args.parent.resolve()
    ):
        parser.error("Output must be a separate sibling of the parent run")
    args.output.mkdir(exist_ok=True)
    with (args.output / ".stage3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, resume, offset = load_run(
            args.parent, args.output, args.target, args.save_every, args.max_rounds
        )
        asyncio.run(run(config, resume, offset))


if __name__ == "__main__":
    main()
