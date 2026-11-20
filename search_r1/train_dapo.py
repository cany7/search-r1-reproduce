"""Stage three: synchronous refill from fresh prompts, one optimizer step per full batch."""

import argparse
import asyncio
import fcntl
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import chz
import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.rollout_logging import serialize_rollout_summaries
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.cost_estimation import CostTracker, TokenUsage, fetch_model_pricing
from tinker_cookbook.utils.misc_utils import all_same

from search_r1.env.search_env import SearchRLDataset
from search_r1.env.tools import RetrievalConfig
from search_r1.skyrl_compat import patch_create_from_state
from search_r1.training.cookbook_bridge import SearchR1DatasetBuilder
from search_r1.wandb_compat import persistent_wandb_run

START_STEP = 100
logger = logging.getLogger(__name__)


@chz.chz
class Stage3Config(train.Config):
    parent_state_path: str
    max_sampling_rounds: int = 8


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


@dataclass
class Batch:
    start: int
    attempted: int = 0
    rounds: int = 0
    constant: int = 0
    surplus: int = 0
    perfect: int = 0
    nonpositive: int = 0
    builders: list = field(default_factory=list)
    groups: list = field(default_factory=list)

    @property
    def offset(self):
        return self.start + self.attempted

    def metrics(self):
        return {
            "sampling/attempted_groups": self.attempted,
            "sampling/selected_groups": len(self.groups),
            "sampling/constant_groups": self.constant,
            "sampling/surplus_groups": self.surplus,
            "sampling/discarded_groups": self.constant + self.surplus,
            "sampling/eligible_fraction": (self.attempted - self.constant) / self.attempted,
            "sampling/all_full_reward_fraction": self.perfect / self.attempted,
            "sampling/all_same_nonpositive_fraction": self.nonpositive / self.attempted,
            "sampling/rounds": self.rounds,
            "sampling/start_prompt_offset": self.start,
            "sampling/next_prompt_offset": self.offset,
        }


async def collect_batch(
    builders, offset, target_size, group_size, max_rounds, rollout_wave, record
):
    """Select in shuffled dataset order. No rollout from the old policy crosses an update."""
    batch = Batch(offset)
    for _ in range(max_rounds):
        need = target_size - len(batch.groups)
        # One wave never exceeds the established concurrency. Estimate the final
        # wave from the observed pass rate to avoid generating a whole extra batch.
        wave_size = target_size
        if batch.groups:
            wave_size = min(
                target_size, max(8, math.ceil(need * batch.attempted / len(batch.groups)))
            )
        wave = builders[batch.offset : batch.offset + wave_size]
        if not wave:
            break
        results = await rollout_wave(wave)
        batch.rounds += 1
        for builder, group in zip(wave, results, strict=True):
            pos = batch.offset
            if group is None:
                raise RuntimeError(f"Rollout failed at prompt offset {pos}")
            rewards = group.get_total_rewards()
            if len(rewards) != group_size or not all(math.isfinite(r) for r in rewards):
                raise ValueError(f"Incomplete group or nonfinite reward at prompt offset {pos}")
            batch.attempted += 1
            if all_same(rewards):
                reason = "constant"
                batch.constant += 1
                batch.perfect += rewards[0] == 1
                batch.nonpositive += rewards[0] <= 0
            elif len(batch.groups) == target_size:
                reason = "surplus"
                batch.surplus += 1
            else:
                reason = "selected"
                batch.groups.append(group)
                batch.builders.append(builder)
            record(builder, group, pos, reason)
        logger.info(
            "Refill round %d: %d/%d selected after %d prompts",
            batch.rounds,
            len(batch.groups),
            target_size,
            batch.attempted,
        )
        if len(batch.groups) == target_size:
            return batch
    raise RuntimeError(
        f"Refill stopped before update: {len(batch.groups)}/{target_size} groups, "
        f"{batch.attempted} prompts, {batch.rounds} rounds, offsets [{offset}, {batch.offset})"
    )


async def train_updates(config, resume, offset, client, manager, log, costs):
    dataset, _ = await config.dataset_builder()
    if not isinstance(dataset, SearchRLDataset) or not 0 <= offset <= len(
        dataset.env_group_builders
    ):
        raise ValueError(f"Invalid stage-three dataset or prompt offset {offset}")
    tokenizer = train.get_tokenizer(config.model_name)
    sampler = await client.save_weights_and_get_sampling_client()
    strategy = config.effective_rollout_strategy()
    for step in range(resume.batch, config.max_steps):
        start = monotonic()
        discarded_before = costs.discarded.sampled
        iteration = Path(config.log_path) / f"iteration_{step:06d}"
        iteration.mkdir(exist_ok=True)
        pending = iteration / "train_rollout_summaries.pending.jsonl"

        async def rollout_wave(wave):
            return await train.gather_with_progress(
                (
                    train.do_group_rollout_and_filter_constant_reward(
                        sampler,
                        builder,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        do_remove_constant_reward_groups=False,
                        enable_logging=False,
                        strategy=strategy,
                        termination=config.effective_termination(),
                    )
                    for builder in wave
                ),
                desc=f"Update {step + 1} refill",
            )

        with trace.trace_iteration(step=step) as window:
            with pending.open("w") as output:

                def record(builder, group, pos, reason):
                    if reason != "selected":
                        train.count_discarded_rollout(costs, group)
                    rows = serialize_rollout_summaries(
                        split="train",
                        iteration=step,
                        trajectory_groups_P=[group],
                        taglist_P=[builder.logging_tags()],
                        sampling_client_steps_P=[step],
                    )
                    for row in rows:
                        row.update(
                            group_idx=pos - offset,
                            prompt_offset=pos,
                            example_id=builder.datum["id"],
                            selection=reason,
                        )
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")

                async with trace.scope_span("sampling"):
                    batch = await collect_batch(
                        dataset.env_group_builders,
                        offset,
                        dataset.batch_size,
                        config.dataset_builder.group_size,
                        config.max_sampling_rounds,
                        rollout_wave,
                        record,
                    )
            sampler, metrics = await train.do_train_step_and_get_sampling_client(
                config,
                step,
                client,
                manager,
                None,
                tokenizer,
                batch.builders,
                batch.groups,
                cost_tracker=costs,
            )

        completed = step + 1
        offset = batch.offset
        pending.replace(iteration / "train_rollout_summaries.jsonl")
        metrics.update(window.get_timing_metrics())
        metrics.update(batch.metrics())
        del batch  # Release full trajectories before starting another refill.
        discarded_tokens = costs.discarded.sampled - discarded_before
        metrics["sampling/discarded_action_tokens"] = discarded_tokens
        metrics["sampling/all_action_tokens"] = costs.step.sampled + discarded_tokens
        metrics.update(
            {
                "progress/batch": step,
                "progress/completed_updates": completed,
                "optim/lr": config.learning_rate,
                "time/total": monotonic() - start,
            }
        )
        log.log_metrics(metrics, step=step)
        state = train._loop_state(completed, costs) | {
            "next_prompt_offset": offset,
            "stage3_run": config.wandb_name,
            "parent_state_path": config.parent_state_path,
        }
        # Commit the cursor together with the actual optimizer state, only after
        # metrics/rollouts have been published. A crash replays from the last pair.
        if completed == config.max_steps:
            await manager.save_final_async(loop_state=state)
        elif config.save_every and completed % config.save_every == 0:
            await manager.save_periodic_async(step=completed, loop_state=state)


async def run(config, resume, offset):
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
        train._drop_metrics_logged_after_checkpoint(log.store, resume.batch)
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
