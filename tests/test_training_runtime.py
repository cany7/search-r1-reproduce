"""Offline integration of local checkpoint runtime with the real local loop."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chz
import pytest
from test_training_loop import Builder, Client, config
from training_cases import group

from search_r1.env.search_env import SearchR1DatasetBuilder, SearchRLDataset
from search_r1.training import runtime, sampling


@pytest.mark.parametrize("mode", ["ordinary", "dynamic"])
def test_base_then_own_resume_reaches_same_local_loop(tmp_path, monkeypatch, mode):
    events, loaded, sampled = [], [], []
    archives = {}
    closed_tools = []

    class SavingClient(Client):
        async def save_state(self, name, **kwargs):
            path = f"tinker://fake/weights/{name}"
            archives[path] = self.step
            return SimpleNamespace(path=path)

        async def save_weights_for_sampler(self, name, **kwargs):
            return SimpleNamespace(path=f"tinker://fake/sampler/{name}")

    class Service:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append(("close_service",))

        async def create_lora_training_client(self, model, **kwargs):
            loaded.append("base")
            return SavingClient(events)

        async def create_training_client_from_state_with_optimizer(self, path, **kwargs):
            loaded.append(path)
            client = SavingClient(events)
            client.step = archives[path]
            return client

    class Tool:
        async def aclose(self):
            closed_tools.append(True)

    async def dataset(self):
        tool = Tool()
        builders = [Builder(i) for i in range(30)]
        for builder in builders:
            builder.search_tool = tool
        return SearchRLDataset(builders, 4), None

    async def sample(sampler, builder, **kwargs):
        i = int(builder.datum["id"])
        sampled.append((sampler, i))
        return group([0, 0] if mode == "dynamic" and i < 3 else [0.1, 0.12])

    monkeypatch.setattr(runtime.tinker, "ServiceClient", Service)
    monkeypatch.setattr(runtime, "fetch_model_pricing", lambda *args: None)
    monkeypatch.setattr(
        runtime.tinker_backend,
        "get_renderer_name_from_checkpoint_async",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(SearchR1DatasetBuilder, "__call__", dataset)
    monkeypatch.setattr(sampling, "sample_group", sample)
    cfg = config(tmp_path, sampling_mode=mode, skyrl_compat=False)
    asyncio.run(runtime.main(cfg))
    asyncio.run(runtime.main(chz.replace(cfg, max_steps=3)))
    assert loaded == ["base", "tinker://fake/weights/final"]
    assert len(closed_tools) == 2  # one shared tool per invocation, not one per builder
    assert [e[1] for e in events if e[0] == "adam"] == [1, 2, 3]
    assert [i for _, i in sampled] == list(range(16 if mode == "dynamic" else 12))
    records = [
        json.loads(line) for line in (tmp_path / "checkpoints.jsonl").read_text().splitlines()
    ]
    assert records[-1]["batch"] == 3
    assert records[-1]["next_prompt_offset"] == (16 if mode == "dynamic" else 12)
    assert records[-1]["name"] == "final"


@pytest.mark.parametrize("mode,batch_size", [("ordinary", 4), ("dynamic", 3)])
def test_failed_save_replays_same_prompts_and_optimizer_from_last_child_step(
    tmp_path, monkeypatch, mode, batch_size
):
    from test_training_checkpoints import write_parent

    from search_r1.training.checkpoints import full_records

    dataset = SearchRLDataset([Builder(i) for i in range(61)], batch_size)
    parent = write_parent(tmp_path / "parent", dataset)
    child = tmp_path / "child"
    cfg = config(
        child,
        parent_run=str(parent),
        parent_step=7,
        max_steps=10,
        batch_size=batch_size,
        sampling_mode=mode,
    )
    loads, samples = [], []
    archives = {"tinker://parent/weights/chosen": 7}
    failed = False

    class Training(Client):
        async def load_state_with_optimizer(self, path):
            loads.append(path)
            self.step = archives[path]

        async def save_state(self, name, ttl_seconds):
            assert ttl_seconds == (None if name == "final" else cfg.ttl_seconds)
            path = f"tinker://model-{len(loads)}/weights/{name}"
            archives[path] = self.step
            return SimpleNamespace(path=path)

        async def save_weights_for_sampler(self, name, ttl_seconds):
            nonlocal failed
            if not failed and self.step == 9:
                failed = True
                raise OSError("sampler save failed")
            return SimpleNamespace(path=f"tinker://model-{len(loads)}/sampler/{name}")

    class Service:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        async def create_lora_training_client(self, model, **kwargs):
            return Training([])

    async def build(self):
        return dataset, None

    async def sample(policy_step, builder, **kwargs):
        pos = int(builder.datum["id"])
        samples.append((len(loads), policy_step, pos))
        return group([1, 1] if mode == "dynamic" and pos % 3 == 0 else [0.1, 0.12])

    monkeypatch.setattr(runtime.tinker, "ServiceClient", Service)
    monkeypatch.setattr(
        runtime.tinker_backend,
        "get_renderer_name_from_checkpoint_async",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(runtime, "fetch_model_pricing", lambda model: None)
    monkeypatch.setattr(SearchR1DatasetBuilder, "__call__", build)
    monkeypatch.setattr(sampling, "sample_group", sample)
    with pytest.raises(OSError, match="sampler save failed"):
        asyncio.run(runtime.main(cfg))

    last = full_records(child)[-1]
    assert (last["owner"], last["batch"]) == ("child", 8)
    assert samples[0] == (1, 7, 28)
    assert (child / "iteration_000008/train_rollout_summaries.jsonl").is_file()
    assert len((child / "metrics.jsonl").read_text().splitlines()) == 2
    asyncio.run(runtime.main(chz.replace(cfg, parent_run=None, parent_step=None)))
    assert loads == ["tinker://parent/weights/chosen", last["state_path"]]
    original = [pos for session, step, pos in samples if session == 1 and step == 8]
    replay = [pos for session, step, pos in samples if session == 2 and step == 8]
    assert original == replay and replay[0] == last["next_prompt_offset"]

    rows = full_records(child)
    assert [row["batch"] for row in rows if row["owner"] == "child"] == [8, 9, 10]
    assert [row["batch"] for row in rows if row["owner"] == "parent"] == (
        [7] if mode == "ordinary" else []
    )
    metrics = [json.loads(line) for line in (child / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [7, 8, 9]
    assert rows[-1]["cumulative_tokens"]["trained"] == metrics[-1]["tokens/cumulative/trained"]
    assert rows[-1]["next_prompt_offset"] == samples[-1][-1] + 1
    # Repeated completion starts no SDK session and commits no duplicate records.
    asyncio.run(runtime.main(cfg))
    assert len(loads) == 2 and rows == full_records(child)
