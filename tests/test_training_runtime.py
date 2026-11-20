"""Offline integration of temporary startup adapter with the real local loop."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chz
import pytest
from test_training_loop import Builder, Client, config
from training_cases import group

from search_r1.env.search_env import SearchR1DatasetBuilder, SearchRLDataset
from search_r1.training import legacy_runtime, sampling


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

    monkeypatch.setattr(legacy_runtime.tinker, "ServiceClient", Service)
    monkeypatch.setattr(legacy_runtime, "fetch_model_pricing", lambda *args: None)
    monkeypatch.setattr(
        legacy_runtime.checkpoint_utils, "check_renderer_name_for_checkpoint_async", AsyncMock()
    )
    monkeypatch.setattr(SearchR1DatasetBuilder, "__call__", dataset)
    monkeypatch.setattr(sampling, "sample_group", sample)
    cfg = config(tmp_path, sampling_mode=mode)
    asyncio.run(legacy_runtime.main(cfg))
    asyncio.run(legacy_runtime.main(chz.replace(cfg, max_steps=3)))
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
