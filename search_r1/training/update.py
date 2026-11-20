# Adapted from tinker-cookbook/rl/train.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified: explicit local update API; synchronous snapshot publication, no framework dispatch.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""Convert selected groups, submit gradients/Adam, and publish the new policy."""

import asyncio
from dataclasses import dataclass, replace

import tinker

from search_r1.training.data import assemble_training_data, compute_advantages
from search_r1.training.metrics import compute_kl_sample_train
from search_r1.training.utils import split_list


def prepare_data(groups):
    data, _ = assemble_training_data(groups, compute_advantages(groups))
    # Initial-overflow trajectories have no action targets. Never send a zero-signal
    # datum to the service. Zero *advantage* groups with real actions remain valid.
    return [d for d in data if any(d.loss_fn_inputs["mask"].data)]


def without_mask(datum):
    # The SDK loss consumes advantages, targets and old logprobs; mask is local metadata.
    return replace(
        datum, loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"}
    )


@dataclass
class UpdateResult:
    sampler: object | None
    data: list
    metrics: dict
    optimizer_steps: int


class Updater:
    """One iteration's ordered SDK submissions; snapshots are published after the last Adam."""

    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.pending = []
        self.data = []
        self.forward_results = []
        self.metrics = {}
        self.optimizer_steps = 0

    def submit(self, awaitable):
        future = asyncio.ensure_future(awaitable)
        self.pending.append(future)
        return future

    def forward(self, data):
        self.data.extend(data)
        future = self.submit(
            self.client.forward_backward(
                [without_mask(d) for d in data],
                loss_fn=self.config.loss_fn,
                loss_fn_config=self.config.loss_fn_config,
            )
        )
        self.forward_results.append(future)
        return future

    def optimize(self):
        self.optimizer_steps += 1
        return self.submit(
            self.client.optim_step(
                tinker.AdamParams(
                    learning_rate=self.config.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )
        )

    async def finish(self):
        if not self.data:
            return UpdateResult(None, [], {}, 0)
        # Submit before waiting on results, preserving the upstream SDK pipeline.
        published = self.submit(
            self.client.save_weights_and_get_sampling_client(
                kind="snapshot",
                record_stability_info=False,
            )
        )
        results = await asyncio.gather(*self.forward_results)
        logprobs = [
            item["logprobs"].to_torch() for result in results for item in result.loss_fn_outputs
        ]
        sampler = await published
        self.metrics.update(compute_kl_sample_train(self.data, logprobs))
        return UpdateResult(sampler, self.data, self.metrics, self.optimizer_steps)

    async def close(self):
        for future in self.pending:
            if not future.done():
                future.cancel()
        await asyncio.gather(*self.pending, return_exceptions=True)


async def update_groups(client, config, groups):
    data = prepare_data(groups)
    updater = Updater(client, config)
    try:
        if data:
            # Same linspace partition as the pinned upstream, including uneven splits.
            batches = split_list(data, min(config.num_substeps, len(data)))
            pairs = []
            for batch in batches:
                pairs.append((updater.forward(batch), updater.optimize()))
            # Queue publication before consuming the already-ordered SDK results.
            finish = asyncio.create_task(updater.finish())
            updater.pending.append(finish)
            for forward, optim in pairs:
                await forward
                result = await optim
                updater.metrics.update(result.metrics or {})
            return await finish
        return await updater.finish()
    finally:
        await updater.close()


async def update_stream(client, config, minibatches):
    """Consume completion-ordered minibatches; one Adam per configured substep."""
    updater = Updater(client, config)
    per_substep = config.stream_minibatch_config.num_minibatches
    pending_forward = []
    count = 0
    try:
        async for groups in minibatches:
            data = prepare_data(groups)
            if data:
                pending_forward.append(updater.forward(data))
            count += 1
            if count % per_substep == 0 and pending_forward:
                optim = updater.optimize()
                await asyncio.gather(*pending_forward)
                result = await optim
                updater.metrics.update(result.metrics or {})
                pending_forward = []
        expected = per_substep * config.num_substeps
        if count != expected:
            raise ValueError(f"Incomplete streaming batch: {count}/{expected} minibatches")
        return await updater.finish()
    finally:
        await updater.close()
