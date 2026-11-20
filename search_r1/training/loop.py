# Adapted from tinker-cookbook/rl/train.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified: one local ordinary/dynamic loop; explicit task cleanup and post-log checkpoint boundary.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""Shared synchronous loop: collect → prepare/update → publish → log/save.

Checkpoint and logger objects are supplied by the temporary runtime adapter.
This module and all computation it invokes are independent of cookbook.
"""

import asyncio
import json
import math
from contextlib import aclosing
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from search_r1.training import sampling
from search_r1.training.data import remove_constant_reward_groups
from search_r1.training.metrics import (
    compute_trajectory_metrics,
    count_discarded_rollout,
    rollout_usage,
    training_usage,
)
from search_r1.training.records import serialize_rollout_summaries
from search_r1.training.update import update_groups, update_stream
from search_r1.training.utils import all_same


def loop_state(step, costs):
    return {
        "batch": step,
        "cumulative_tokens": asdict(costs.cumulative),
        "discarded_tokens": asdict(costs.discarded),
    }


async def run_updates(
    config,
    dataset,
    start_step,
    end_step,
    offset,
    client,
    manager,
    log,
    costs,
    checkpoint_fields=None,
):
    """Run both sampling modes; a dynamic tail raises before any optimizer call."""
    config.validate()
    if not 0 <= offset <= len(dataset.env_group_builders):
        raise ValueError(f"Invalid prompt offset {offset}")
    if start_step >= end_step:
        return
    sampler = await client.save_weights_and_get_sampling_client()
    for step in range(start_step, end_step):
        sampler, offset, state = await _run_iteration(
            config,
            dataset,
            step,
            end_step,
            offset,
            sampler,
            client,
            log,
            costs,
        )
        state.update(checkpoint_fields or {})
        # All iteration trajectories/datums leave scope before the next collection.
        # Full state follows published records; stage 5 adds ownership/fork validation.
        if step + 1 == end_step:
            await manager.save_final_async(loop_state=state)
        elif config.save_every and (step + 1) % config.save_every == 0:
            await manager.save_periodic_async(step=step + 1, loop_state=state)


async def _run_iteration(config, dataset, step, end_step, offset, sampler, client, log, costs):
    start = monotonic()
    discarded_before = costs.discarded.sampled
    iteration = Path(config.log_path) / f"iteration_{step:06d}"
    iteration.mkdir(parents=True, exist_ok=True)
    pending = iteration / "train_rollout_summaries.pending.jsonl"
    selected_builders, selected_groups = [], []
    batch_metrics = {}

    async def one(builder):
        group = await sampling.sample_group(
            sampler, builder, max_tokens=config.max_tokens, temperature=config.temperature
        )
        rewards = group.get_total_rewards()
        if len(rewards) != config.dataset_builder.group_size or not all(
            math.isfinite(r) for r in rewards
        ):
            raise ValueError("Incomplete group or nonfinite reward")
        return group

    async def wave(builders):
        return await sampling.gather_rollouts(one(builder) for builder in builders)

    with pending.open("w") as output:

        def record(builder, group, pos, reason):
            if reason != "selected":
                count_discarded_rollout(costs, group)
            if not config.rollout_json_export:
                return
            rows = serialize_rollout_summaries(
                split="train",
                iteration=step,
                trajectory_groups_P=[group],
                taglist_P=[builder.logging_tags()],
                sampling_client_steps_P=[step],
            )
            for row in rows:
                row["group_idx"] = pos - offset
                if config.sampling_mode == "dynamic":
                    row.update(prompt_offset=pos, example_id=builder.datum["id"], selection=reason)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

        if config.sampling_mode == "dynamic":
            batch = await sampling.collect_batch(
                dataset.env_group_builders,
                offset,
                dataset.batch_size,
                config.dataset_builder.group_size,
                config.max_sampling_rounds,
                wave,
                record,
            )
            selected_builders, selected_groups = batch.builders, batch.groups
            next_offset = batch.offset
            batch_metrics = batch.metrics()
            result = await update_groups(client, config, selected_groups)
        else:
            builders = list(dataset.get_batch(step % len(dataset)))
            if len(builders) != dataset.batch_size:
                raise ValueError("Ordinary sampling requires a complete fixed batch")
            next_offset = ((step % len(dataset)) + 1) * dataset.batch_size
            if config.stream_minibatch_config is None:
                groups = await wave(builders)
                kept = (
                    remove_constant_reward_groups(groups)
                    if config.remove_constant_reward_groups
                    else groups
                )
                keep_ids = {id(g) for g in kept}
                for pos, (builder, group) in enumerate(zip(builders, groups, strict=True), offset):
                    record(builder, group, pos, "selected" if id(group) in keep_ids else "constant")
                    if id(group) in keep_ids:
                        selected_builders.append(builder)
                        selected_groups.append(group)
                result = await update_groups(client, config, selected_groups)
            else:
                per_minibatch = dataset.batch_size // (
                    config.num_substeps * config.stream_minibatch_config.num_minibatches
                )

                async def completed(index, builder):
                    return index, builder, await one(builder)

                async def minibatches():
                    tasks = [asyncio.create_task(completed(i, b)) for i, b in enumerate(builders)]
                    chunk, received = [], 0
                    try:
                        for future in asyncio.as_completed(tasks):
                            pos, builder, group = await future
                            keep = not config.remove_constant_reward_groups or not all_same(
                                group.get_total_rewards()
                            )
                            record(builder, group, offset + pos, "selected" if keep else "constant")
                            if keep:
                                chunk.append(group)
                                selected_groups.append(group)
                                selected_builders.append(builder)
                            received += 1
                            if received % per_minibatch == 0:
                                yield chunk
                                chunk = []
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

                async with aclosing(minibatches()) as stream:
                    result = await update_stream(client, config, stream)
    if result.sampler is not None:
        sampler = result.sampler
    offset = next_offset
    metrics = compute_trajectory_metrics(
        selected_groups, [b.logging_tags() for b in selected_builders]
    )
    metrics.update(result.metrics)
    metrics.update(costs.record_step(rollout_usage(selected_groups) + training_usage(result.data)))
    metrics.update(batch_metrics)
    discarded_tokens = costs.discarded.sampled - discarded_before
    metrics.update(
        {
            "progress/batch": step,
            "progress/completed_updates": step + 1,
            "progress/done_frac": (step + 1) / end_step,
            "optim/lr": config.learning_rate,
            "optim/optimizer_steps": result.optimizer_steps,
            "time/total": monotonic() - start,
            "sampling/discarded_action_tokens": discarded_tokens,
            "sampling/all_action_tokens": costs.step.sampled + discarded_tokens,
        }
    )
    if config.rollout_json_export:
        pending.replace(iteration / "train_rollout_summaries.jsonl")
    else:
        pending.unlink()
    log.log_metrics(metrics, step=step)
    state = loop_state(step + 1, costs) | {"next_prompt_offset": offset}
    return sampler, offset, state
