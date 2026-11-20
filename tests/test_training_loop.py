"""Only local fake SDK calls. No models, service clients, retrieval or training jobs."""

import asyncio
import json
from types import SimpleNamespace

import chz
import pytest
from training_cases import cases, group

from search_r1.config import CLIConfig
from search_r1.env.search_env import SearchRLDataset
from search_r1.train import build_config
from search_r1.training import sampling
from search_r1.training.loop import run_updates
from search_r1.training.update import update_groups
from search_r1.training.usage import CostTracker


class Builder:
    def __init__(self, index):
        self.datum = {"id": str(index)}

    def logging_tags(self):
        return ["nq"]


class Ready:
    """The SDK submits on call; waiting only consumes a result."""

    def __init__(self, result):
        self.result = result

    def __await__(self):
        async def ready():
            return self.result

        return ready().__await__()


class Client:
    def __init__(self, events):
        self.events = events
        self.step = 0
        self.data = []
        self.params = []
        self.on_forward = None

    def forward_backward(self, data, loss_fn, loss_fn_config):
        self.events.append(("forward", len(data), loss_fn, loss_fn_config))
        self.data.extend(data)
        assert all("mask" not in d.loss_fn_inputs for d in data)
        if self.on_forward:
            self.on_forward()
        return Ready(
            SimpleNamespace(
                loss_fn_outputs=[{"logprobs": d.loss_fn_inputs["logprobs"]} for d in data]
            )
        )

    def optim_step(self, params):
        self.step += 1
        self.params.append(params)
        self.events.append(("adam", self.step))
        return Ready(SimpleNamespace(metrics={"fake/last_optimizer": self.step}))

    def save_weights_and_get_sampling_client(self, **kwargs):
        self.events.append(("publish", self.step))
        return Ready(self.step)


class Manager:
    def __init__(self, events):
        self.events = events
        self.states = []

    async def save_final_async(self, loop_state):
        self.events.append(("final", loop_state["batch"]))
        self.states.append(loop_state)

    async def save_periodic_async(self, step, loop_state):
        self.events.append(("save", step))
        self.states.append(loop_state)


class Log:
    def __init__(self, events):
        self.events = events
        self.rows = []

    def log_metrics(self, metrics, step):
        self.events.append(("log", step))
        self.rows.append(metrics)


def config(tmp_path, **kwargs):
    options = {
        "batch_size": 4,
        "group_size": 2,
        "seed": 19,
        "learning_rate": 0.0003,
        "log_path": str(tmp_path),
        "max_steps": 2,
        "save_every": 1,
        "wandb_project": None,
    }
    options.update(kwargs)
    return build_config(CLIConfig(**options))


def setup_loop(tmp_path, **kwargs):
    cfg = config(tmp_path, **kwargs)
    events = []
    return cfg, events, Client(events), Manager(events), Log(events), CostTracker()


def test_ordinary_two_updates_use_new_policy_and_commit_after_logs(tmp_path, monkeypatch):
    cfg, events, client, manager, log, costs = setup_loop(tmp_path)
    seen = []

    async def sample(sampler, builder, **kwargs):
        seen.append((sampler, int(builder.datum["id"])))
        return group([0.1, 0.12])

    monkeypatch.setattr(sampling, "sample_group", sample)
    dataset = SearchRLDataset([Builder(i) for i in range(9)], 4)
    asyncio.run(run_updates(cfg, dataset, 0, 2, 0, client, manager, log, costs))
    assert seen == [(0, i) for i in range(4)] + [(1, i) for i in range(4, 8)]
    assert [event[0] for event in events] == [
        "publish",
        "forward",
        "adam",
        "publish",
        "log",
        "save",
        "forward",
        "adam",
        "publish",
        "log",
        "final",
    ]
    assert [s["next_prompt_offset"] for s in manager.states] == [4, 8]
    assert costs.cumulative.trained == 2 * 4 * 2 * 3
    assert all(
        (p.learning_rate, p.beta1, p.beta2, p.eps) == (0.0003, 0.9, 0.95, 1e-8)
        for p in client.params
    )
    assert len(list(tmp_path.glob("*/train_rollout_summaries.jsonl"))) == 2
    assert not list(tmp_path.glob("*/*.pending.jsonl"))


def test_dynamic_refill_uses_same_policy_and_counts_surplus(tmp_path, monkeypatch):
    cfg, events, client, manager, log, costs = setup_loop(tmp_path, sampling_mode="dynamic")
    seen = []

    async def sample(sampler, builder, **kwargs):
        i = int(builder.datum["id"])
        seen.append((sampler, i))
        return group([0.2, 0.2] if i < 3 else [0.1, 0.12])

    monkeypatch.setattr(sampling, "sample_group", sample)
    dataset = SearchRLDataset([Builder(i) for i in range(20)], 4)
    asyncio.run(run_updates(cfg, dataset, 0, 2, 0, client, manager, log, costs))
    assert seen == [(0, i) for i in range(8)] + [(1, i) for i in range(8, 12)]
    assert [s["next_prompt_offset"] for s in manager.states] == [8, 12]
    assert (log.rows[0]["sampling/constant_groups"], log.rows[0]["sampling/surplus_groups"]) == (
        3,
        1,
    )
    assert costs.discarded.sampled == 4 * 2 * 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "iteration_000000/train_rollout_summaries.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[-1]["selection"] == "surplus" and rows[-1]["prompt_offset"] == 7
    assert len(client.params) == 2


@pytest.mark.parametrize("size,rounds", [(30, 1), (3, 8)])
def test_dynamic_incomplete_batch_never_updates_or_saves(tmp_path, monkeypatch, size, rounds):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, sampling_mode="dynamic", max_sampling_rounds=rounds
    )

    async def sample(*args, **kwargs):
        return group([0.5, 0.5])

    monkeypatch.setattr(sampling, "sample_group", sample)
    with pytest.raises(RuntimeError, match="Refill stopped before update"):
        asyncio.run(
            run_updates(
                cfg,
                SearchRLDataset([Builder(i) for i in range(size)], 4),
                0,
                1,
                0,
                client,
                manager,
                log,
                costs,
            )
        )
    assert events == [("publish", 0)]
    assert not manager.states and not log.rows


@pytest.mark.parametrize("substeps,minibatches", [(1, 2), (2, 2)])
def test_streaming_starts_forward_before_slow_rollouts_finish(
    tmp_path, monkeypatch, substeps, minibatches
):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, stream_minibatch=True, num_minibatches=minibatches, num_substeps=substeps
    )
    seen = []

    async def scenario():
        first_forward = asyncio.Event()
        client.on_forward = first_forward.set
        per_minibatch = 4 // (substeps * minibatches)

        async def sample(sampler, builder, **kwargs):
            i = int(builder.datum["id"])
            if i >= per_minibatch:
                await (
                    first_forward.wait()
                )  # Deadlocks if implementation gathers the full batch first.
            seen.append((sampler, i))
            return group([0, 1])

        monkeypatch.setattr(sampling, "sample_group", sample)
        await asyncio.wait_for(
            run_updates(
                cfg,
                SearchRLDataset([Builder(i) for i in range(4)], 4),
                0,
                1,
                0,
                client,
                manager,
                log,
                costs,
            ),
            timeout=2,
        )

    asyncio.run(scenario())
    assert all(sampler == 0 for sampler, _ in seen)
    assert len([e for e in events if e[0] == "forward"]) == substeps * minibatches
    assert len(client.params) == substeps
    assert [e for e in events if e[0] == "publish"] == [("publish", 0), ("publish", substeps)]
    assert log.rows[0]["env/all/total_episodes"] == 8


def test_stream_failure_cancels_pending_rollouts_and_does_not_checkpoint(tmp_path, monkeypatch):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, stream_minibatch=True, num_minibatches=2
    )
    drained = []

    async def scenario():
        started = asyncio.Event()
        count = 0

        async def sample(sampler, builder, **kwargs):
            nonlocal count
            count += 1
            if count == 4:
                started.set()
            try:
                await started.wait()
                if builder.datum["id"] == "0":
                    raise RuntimeError("broken rollout")
                await asyncio.Event().wait()
            finally:
                drained.append(builder.datum["id"])

        monkeypatch.setattr(sampling, "sample_group", sample)
        with pytest.raises(RuntimeError, match="broken rollout"):
            await asyncio.wait_for(
                run_updates(
                    cfg,
                    SearchRLDataset([Builder(i) for i in range(4)], 4),
                    0,
                    1,
                    0,
                    client,
                    manager,
                    log,
                    costs,
                ),
                2,
            )
        assert sorted(drained) == ["0", "1", "2", "3"]

    asyncio.run(scenario())
    assert not manager.states and not client.params


@pytest.mark.parametrize("stream", [False, True])
def test_zero_action_batch_never_sends_empty_data(tmp_path, monkeypatch, stream):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, stream_minibatch=stream, num_minibatches=2
    )

    async def sample(*args, **kwargs):
        return cases()["all_overflow"][0]

    monkeypatch.setattr(sampling, "sample_group", sample)
    asyncio.run(
        run_updates(
            cfg,
            SearchRLDataset([Builder(i) for i in range(4)], 4),
            0,
            1,
            0,
            client,
            manager,
            log,
            costs,
        )
    )
    assert not client.data and not client.params
    assert log.rows[0]["optim/optimizer_steps"] == 0


@pytest.mark.parametrize("loss", ["importance_sampling", "ppo", "cispo"])
def test_substeps_loss_parameters_and_snapshot(tmp_path, loss):
    cfg = chz.replace(
        config(tmp_path), num_substeps=3, loss_fn=loss, loss_fn_config={"clip_low_threshold": 0.2}
    )
    events = []
    client = Client(events)
    result = asyncio.run(update_groups(client, cfg, cases()["centered_partial_rewards"]))
    assert [e[1] for e in events if e[0] == "forward"] == [1, 2, 2]  # pinned linspace split
    assert all(e[2:] == (loss, cfg.loss_fn_config) for e in events if e[0] == "forward")
    assert result.optimizer_steps == 3 and result.sampler == 3
    assert result.metrics["optim/kl_sample_train_v1"] == 0
    assert result.metrics["fake/last_optimizer"] == 3
    assert events[-1] == ("publish", 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_mode": "dynamic", "stream_minibatch": True},
        {"sampling_mode": "dynamic", "num_substeps": 2},
        {"stream_minibatch": True, "num_minibatches": 3},
        {"stream_minibatch": True, "num_minibatches": 0},
        {"batch_size": 0},
        {"group_size": 0},
        {"max_sampling_rounds": 0},
    ],
)
def test_invalid_cli_modes_rejected_before_runtime(tmp_path, kwargs):
    with pytest.raises(ValueError):
        config(tmp_path, **kwargs)


def test_four_recipe_configs_preserve_defaults(tmp_path):
    assert CLIConfig().batch_size == 512
    base = config(tmp_path)
    limited = config(tmp_path, limit_search=True)
    penalized = config(tmp_path, limit_search=True, max_turns_unanswered_reward=-0.3)
    dynamic = config(tmp_path, limit_search=True, sampling_mode="dynamic")
    assert not base.dataset_builder.limit_search
    assert limited.dataset_builder.limit_search
    assert penalized.dataset_builder.max_turns_unanswered_reward == -0.3
    assert dynamic.sampling_mode == "dynamic" and dynamic.remove_constant_reward_groups


@pytest.mark.parametrize("stream", [False, True])
def test_cancel_drains_all_sampling_tasks(tmp_path, monkeypatch, stream):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, stream_minibatch=stream, num_minibatches=2
    )
    drained = []

    async def scenario():
        started = asyncio.Event()
        count = 0

        async def sample(sampler, builder, **kwargs):
            nonlocal count
            count += 1
            if count == 4:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.append(builder.datum["id"])

        monkeypatch.setattr(sampling, "sample_group", sample)
        task = asyncio.create_task(
            run_updates(
                cfg,
                SearchRLDataset([Builder(i) for i in range(4)], 4),
                0,
                1,
                0,
                client,
                manager,
                log,
                costs,
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(drained) == ["0", "1", "2", "3"]

    asyncio.run(scenario())
    assert not manager.states and not client.params


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["forward", "adam", "publish", "log"])
def test_update_failure_never_commits_checkpoint(tmp_path, monkeypatch, stream, failure):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, stream_minibatch=stream, num_minibatches=2
    )

    async def sample(*args, **kwargs):
        return group([0.1, 0.12])

    monkeypatch.setattr(sampling, "sample_group", sample)

    async def fail(*args, **kwargs):
        raise RuntimeError("failed " + failure)

    if failure == "forward":
        client.forward_backward = fail
    elif failure == "adam":
        client.optim_step = fail
    elif failure == "publish":

        def publish(**kwargs):
            return Ready(0) if not kwargs else fail()

        client.save_weights_and_get_sampling_client = publish
    else:

        def fail_log(*args, **kwargs):
            raise RuntimeError("failed log")

        log.log_metrics = fail_log
    with pytest.raises(RuntimeError, match="failed " + failure):
        asyncio.run(
            run_updates(
                cfg,
                SearchRLDataset([Builder(i) for i in range(4)], 4),
                0,
                1,
                0,
                client,
                manager,
                log,
                costs,
            )
        )
    assert not manager.states


@pytest.mark.parametrize("stream", [False, True])
def test_constant_filtering_preserves_ordinary_mode_semantics(tmp_path, monkeypatch, stream):
    cfg, events, client, manager, log, costs = setup_loop(
        tmp_path, stream_minibatch=stream, num_minibatches=2
    )
    cfg = chz.replace(cfg, remove_constant_reward_groups=True)

    async def sample(*args, **kwargs):
        return group([0.5, 0.5])

    monkeypatch.setattr(sampling, "sample_group", sample)
    asyncio.run(
        run_updates(
            cfg,
            SearchRLDataset([Builder(i) for i in range(4)], 4),
            0,
            1,
            0,
            client,
            manager,
            log,
            costs,
        )
    )
    # Upstream ordinary sync keeps one zero-advantage group, while streaming drops all.
    assert len(client.params) == (0 if stream else 1)
    assert costs.discarded.sampled == (4 if stream else 3) * 2 * 2


def test_group_rollout_grades_local_environments_and_cleans_up():
    import tinker

    from search_r1.backends.completers import TokensWithLogprobs
    from search_r1.env.types import Env, StepResult

    cleaned = []

    class Answer(Env):
        async def initial_observation(self):
            return tinker.ModelInput.from_ints([1, 2]), [99]

        async def step(self, action, *, extra=None):
            assert action == [3] and extra["stop_reason"] == "stop"
            return StepResult(0.25, True, tinker.ModelInput.empty(), [99])

    class Group:
        async def make_envs(self):
            return [Answer(), Answer()]

        async def compute_group_rewards(self, trajectories, envs):
            return [(0.5, {"graded": 1}), (-0.25, {"graded": 1})]

        async def cleanup(self):
            cleaned.append(True)

    async def policy(*args):
        return TokensWithLogprobs([3], [-0.5])

    result = asyncio.run(sampling.do_group_rollout(Group(), policy))
    assert result.get_total_rewards() == [0.75, 0.0]
    assert cleaned == [True]


def test_group_failure_drains_episodes_before_cleanup(monkeypatch):
    completed = []

    class Group:
        async def make_envs(self):
            return [0, 1]

        async def cleanup(self):
            assert sorted(completed) == [0, 1]
            completed.append("cleanup")

    async def scenario():
        started = asyncio.Event()

        async def episode(policy, env):
            try:
                if env == 0:
                    await started.wait()
                    raise RuntimeError("episode failed")
                started.set()
                await asyncio.Event().wait()
            finally:
                completed.append(env)

        monkeypatch.setattr(sampling, "do_single_rollout", episode)
        with pytest.raises(RuntimeError, match="episode failed"):
            await sampling.do_group_rollout(Group(), None)

    asyncio.run(scenario())
    assert completed[-1] == "cleanup"
