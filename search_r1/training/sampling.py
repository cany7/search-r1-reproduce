# Group rollout adapted from tinker-cookbook/rl/rollouts.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified: local fail-fast tasks and cleanup. Dynamic refill moved from this project's earlier dynamic-sampling CLI.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""Fixed batches or same-policy dynamic refill, in shuffled dataset order."""

import asyncio
import logging
import math
from dataclasses import dataclass, field

from search_r1.backends.completers import TinkerTokenCompleter
from search_r1.env.rollout import do_single_rollout
from search_r1.env.types import TrajectoryGroup
from search_r1.training.utils import all_same

logger = logging.getLogger(__name__)


async def gather_rollouts(awaitables):
    tasks = [asyncio.ensure_future(item) for item in awaitables]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def do_group_rollout(builder, policy):
    """Fail fast, drain sibling episodes, then release group resources."""
    try:
        envs = await builder.make_envs()
        if not envs:
            raise ValueError("Empty environment group")
        trajectories = await gather_rollouts(do_single_rollout(policy, env) for env in envs)
        graded = await builder.compute_group_rewards(trajectories, envs)
        if len(graded) != len(trajectories):
            raise ValueError("Group grading must return one reward per trajectory")
        rewards, metrics = zip(*graded, strict=True)
        return TrajectoryGroup(trajectories, list(rewards), list(metrics))
    finally:
        cleanup = asyncio.create_task(builder.cleanup())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError


async def sample_group(sampler, builder, *, max_tokens, temperature):
    policy = TinkerTokenCompleter(sampler, max_tokens=max_tokens, temperature=temperature)
    return await do_group_rollout(builder, policy)


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
